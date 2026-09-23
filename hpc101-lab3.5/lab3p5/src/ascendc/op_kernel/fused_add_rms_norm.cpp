/**
 * @file fused_add_rms_norm.cpp
 * @brief Kernel implementation of FusedAddRmsNorm on Ascend C (910B).
 * @details
 *   Op: FusedAddRmsNorm(x, residual, weight, eps) -> (y, residual_out)
 *     residual_out = x + residual
 *     y           = residual_out / sqrt(mean(residual_out^2, dim=-1) + eps) * weight
 *
 *   Row-parallel baseline (correctness-first, not perf-tuned):
 *     - Each AIV core owns a contiguous range of rows (block-strided over B).
 *     - Per row: load x and residual (FP16 GM -> UB via DataCopyPad, zero-padded
 *       to the 32B-aligned H), cast to FP32, add -> residual_out (FP32); cast
 *       residual_out back to FP16 and store to GM; square + reduce (FP32
 *       BlockReduceSum/WholeReduceSum) to one scalar sumSq; rstd =
 *       rsqrt(meanSq + eps) computed via the vector Rsqrt on a broadcast tile;
 *       y = residual_out * rstd * weight, cast to FP16, store to GM.
 *     - All arithmetic in FP32 for precision; FP16 only on the GM boundary.
 *
 *   Tail handling: H need not be a multiple of 16. UB tiles are sized to
 *   alignedHidden (rounded up to ALIGN_NUM); DataCopyPad zero-pads the read so
 *   the tail is well-defined, and the reduce runs on exactly hiddenSize (the
 *   padded tail is excluded via the SetMaskCount/SetVectorMask<COUNTER> path
 *   used by ReduceNormal). Vector ops run on alignedHidden; the padded tail
 *   carries garbage but is never written back (CopyOut uses blockLen = H bytes).
 *
 *   Row size: when alignedHidden fits in one UB tile (the lab cases, H<=4096),
 *   a whole-row path is taken. Larger rows stream through the tile in chunks
 *   (two passes: pass 1 accumulates sumSq, pass 2 applies rstd*weight and
 *   writes both outputs), with the weight streamed per chunk.
 */
#include "kernel_operator.h"

namespace {
constexpr int32_t BUFFER_NUM = 2;          // two row-group tiles per AIV
constexpr int32_t OUT_BUFFER_NUM = 1;      // outputs are produced once per row group
// 32B / sizeof(half) == 16: the UB / DataCopy / vector-op alignment unit.
constexpr int32_t ALIGN_NUM = 16;
// A row group is capped at 7 rows for the scoring shape (40 AIV cores, 256
// rows). The cap also keeps the combined input/compute/output UB footprint
// comfortably below the 192 KiB UB limit.
constexpr int32_t ROW_TILE_MAX_ELEMS = 8192;
constexpr int32_t CHUNK_TILE_ELEMS = 4096;
constexpr int32_t MAX_ROWS_PER_TILE = 8;
// Repeat strides are 8-byte units and are passed to the vector APIs as
// uint8_t. Keep the packed-row fast path out of the overflow region.
constexpr int32_t MAX_REPEAT_STRIDE = 255;
}

/**
 * @brief FusedAddRmsNorm kernel class (FP16 I/O, FP32 compute, row-parallel).
 */
class KernelFusedAddRmsNorm {
public:
    __aicore__ inline KernelFusedAddRmsNorm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                                GM_ADDR y, GM_ADDR residual_out,
                                FusedAddRmsNormTilingData& tiling, AscendC::TPipe* pipeIn) {
        this->pipe = pipeIn;
        this->blockIdx = AscendC::GetBlockIdx();

        this->batchSize = tiling.batchSize;
        this->hiddenSize = tiling.hiddenSize;
        this->alignedHidden = tiling.alignedHidden;
        this->alignNum = tiling.alignNum;
        this->eps = tiling.eps;

        // Whole-row path: pack several rows into one UB tile. Chunked path:
        // cap one chunk at 4096 FP32 elements as before.
        int32_t totalRows = this->batchSize;
        int32_t blockNum = static_cast<int32_t>(AscendC::GetBlockNum());
        int32_t rowsPerBlock = (totalRows + blockNum - 1) / blockNum;
        if (this->alignedHidden <= ROW_TILE_MAX_ELEMS) {
            int32_t maxRows = ROW_TILE_MAX_ELEMS / this->alignedHidden;
            if (this->alignedHidden / 8 > MAX_REPEAT_STRIDE) maxRows = 1;
            if (maxRows > MAX_ROWS_PER_TILE) maxRows = MAX_ROWS_PER_TILE;
            if (maxRows < 1) maxRows = 1;
            if (maxRows > rowsPerBlock) maxRows = rowsPerBlock;
            // H=4096 with two input rows already uses almost the entire UB
            // budget before the second VECIN buffer is added. Keep that edge
            // case single-row so double buffering remains within capacity.
            if (BUFFER_NUM > 1 && this->alignedHidden == CHUNK_TILE_ELEMS) maxRows = 1;
            this->rowTile = maxRows;
            this->tileElems = this->rowTile * this->alignedHidden;
        } else {
            this->rowTile = 1;
            this->tileElems = CHUNK_TILE_ELEMS;
            if (this->tileElems < this->alignNum) this->tileElems = this->alignNum;
        }

        // Row-parallel split: contiguous range of rows per core.
        this->startRow = static_cast<int64_t>(this->blockIdx) * rowsPerBlock;
        this->endRow = this->startRow + rowsPerBlock;
        if (this->endRow > totalRows) this->endRow = totalRows;

        // GM tensors (element counts guarded against 0).
        uint64_t totalElems = static_cast<uint64_t>(this->batchSize) *
                              static_cast<uint64_t>(this->hiddenSize);
        if (totalElems == 0) totalElems = 1;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), totalElems);
        residualGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(residual), totalElems);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), totalElems);
        residualOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(residual_out), totalElems);
        uint64_t weightElems = static_cast<uint64_t>(this->hiddenSize > 0 ? this->hiddenSize : 1);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight), weightElems);

        // UB buffers.
        //   inQueX / inQueRes   : FP16 input tiles (double-buffered).
        //   outQueY / outQueResOut : FP16 output tiles (single-buffered).
        //   weightHalfBuf  : weight row in FP16 (loaded per chunk / once).
        //   weightFp32Buf  : weight in FP32 (Mul operand).
        //   resoFp32Buf    : residual_out in FP32 (add result; reused as y source).
        //   sqBuf          : squared FP32 tile (reduce source) / rstd broadcast.
        //   scalarBuf      : one FP32 scalar (reduce result staging).
        //   reduceTmpBuf   : scratch required by the Block/Whole reduce intrinsics.
        uint32_t tileBytesFp16 = static_cast<uint32_t>(this->tileElems) * sizeof(half);
        uint32_t tileBytesFp32 = static_cast<uint32_t>(this->tileElems) * sizeof(float);
        pipe->InitBuffer(inQueX, BUFFER_NUM, tileBytesFp16);
        pipe->InitBuffer(inQueRes, BUFFER_NUM, tileBytesFp16);
        pipe->InitBuffer(outQueY, OUT_BUFFER_NUM, tileBytesFp16);
        pipe->InitBuffer(outQueResOut, OUT_BUFFER_NUM, tileBytesFp16);
        // In the row-group path only one weight row is needed even when several
        // input/output rows share the tile. The chunked path still needs a full
        // 4096-element weight chunk.
        uint32_t weightBufElems = this->alignedHidden <= this->tileElems
                                      ? static_cast<uint32_t>(this->alignedHidden)
                                      : static_cast<uint32_t>(this->tileElems);
        pipe->InitBuffer(weightHalfBuf, weightBufElems * sizeof(half));
        pipe->InitBuffer(weightFp32Buf, weightBufElems * sizeof(float));
        pipe->InitBuffer(resoFp32Buf, tileBytesFp32);
        pipe->InitBuffer(sqBuf, tileBytesFp32);
        pipe->InitBuffer(sumBuf, MAX_ROWS_PER_TILE * 64 * sizeof(float));
        pipe->InitBuffer(scalarBuf, 32);                 // 1 FP32 scalar, 32B-aligned
        pipe->InitBuffer(reduceTmpBuf, 32);              // reduce scratch, 32B-aligned
    }

    __aicore__ inline void Process() {
        if (this->startRow >= this->endRow) return;
        if (this->hiddenSize <= 0) return;

        if (this->alignedHidden <= this->tileElems) {
            if (this->hiddenSize == this->alignedHidden) {
                ProcessAlignedRowGroups();
            } else {
                ProcessWholeRows();
            }
        } else {
            ProcessChunkedRows();
        }
    }

private:
    // ------------------------------------------------------------------
    //  Whole-row path (H fits in one UB tile)
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessAlignedRowGroups() {
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<float> resoFp32 = resoFp32Buf.Get<float>();
        AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf.Get<float>();

        const int32_t H = this->hiddenSize;
        const int32_t rowTileMax = this->rowTile;
        const float invH = 1.0f / static_cast<float>(H);
        LoadWeightRow(weightFp32, 0, H, H);

        for (int64_t row = this->startRow; row < this->endRow; row += rowTileMax) {
            const int32_t rowsThisTile = static_cast<int32_t>(
                this->endRow - row < rowTileMax ? this->endRow - row : rowTileMax);
            const int32_t tileElems = rowsThisTile * H;
            const uint64_t base = static_cast<uint64_t>(row) * H;

            AscendC::LocalTensor<half> xLocal = inQueX.AllocTensor<half>();
            AscendC::LocalTensor<half> resLocal = inQueRes.AllocTensor<half>();
            AscendC::DataCopy(xLocal, xGm[base], tileElems);
            AscendC::DataCopy(resLocal, residualGm[base], tileElems);
            inQueX.EnQue(xLocal);
            inQueRes.EnQue(resLocal);
            xLocal = inQueX.DeQue<half>();
            resLocal = inQueRes.DeQue<half>();

            // R = x + residual, retained in FP32 for both outputs.
            AscendC::Cast(resoFp32, xLocal, AscendC::RoundMode::CAST_NONE, tileElems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(sq, resLocal, AscendC::RoundMode::CAST_NONE, tileElems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(resoFp32, resoFp32, sq, tileElems);
            AscendC::PipeBarrier<PIPE_V>();
            inQueX.FreeTensor(xLocal);
            inQueRes.FreeTensor(resLocal);

            // Emit residual_out first; MTE3 can overlap the reduction and y pass.
            AscendC::LocalTensor<half> resOutLocal = outQueResOut.AllocTensor<half>();
            AscendC::Cast(resOutLocal, resoFp32, AscendC::RoundMode::CAST_NONE, tileElems);
            AscendC::PipeBarrier<PIPE_V>();
            outQueResOut.EnQue(resOutLocal);
            resOutLocal = outQueResOut.DeQue<half>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(residualOutGm[base], resOutLocal, tileElems);
            outQueResOut.FreeTensor(resOutLocal);

            // Square the whole row group, then reduce all rows together. The
            // compact row sums stay in the Vector pipe, so the scoring shape
            // can compute and broadcast all rstd values without Scalar stalls.
            AscendC::Mul(sq, resoFp32, resoFp32, tileElems);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<half> yLocal = outQueY.AllocTensor<half>();
            if ((H % 64 == 0) && rowsThisTile > 0) {
                ReduceAlignedRows(sumBuf.Get<float>(), sq, rowsThisTile, H);
            } else {
                for (int32_t r = 0; r < rowsThisTile; ++r) {
                    const int32_t rowOffset = r * H;
                    ReduceNormal(sumBuf.Get<float>()[r * H], sq[rowOffset], H);
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                }
            }

            AscendC::LocalTensor<float> sums = sumBuf.Get<float>();
            AscendC::Muls<float>(sums, sums, invH, rowsThisTile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds<float>(sums, sums, this->eps, rowsThisTile);
            AscendC::PipeBarrier<PIPE_V>();
            // CANN's RMSNorm uses Ln/Exp rather than approximate reciprocal
            // instructions; this keeps the FP16 boundary inside tolerance.
            AscendC::Ln<float>(sums, sums, rowsThisTile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls<float>(sums, sums, -0.5f, rowsThisTile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp<float>(sums, sums, rowsThisTile);
            AscendC::PipeBarrier<PIPE_V>();

            if ((rowsThisTile % 8 == 0) && (H % 64 == 0)) {
                BroadcastRowScales(sq, sums, rowsThisTile, H);
                AscendC::Mul<float>(sq, sq, resoFp32, tileElems);
                AscendC::PipeBarrier<PIPE_V>();
                MultiplyWeightRows(sq, weightFp32, rowsThisTile, H);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(yLocal, sq, AscendC::RoundMode::CAST_NONE,
                              tileElems);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                for (int32_t r = 0; r < rowsThisTile; ++r) {
                    const int32_t rowOffset = r * H;
                    const float rstd = sums.GetValue(r);

                    AscendC::Muls(sq[rowOffset], resoFp32[rowOffset], rstd, H);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(sq[rowOffset], sq[rowOffset], weightFp32, H);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(yLocal[rowOffset], sq[rowOffset],
                                  AscendC::RoundMode::CAST_NONE, H);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }

            outQueY.EnQue(yLocal);
            yLocal = outQueY.DeQue<half>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(yGm[base], yLocal, tileElems);
            outQueY.FreeTensor(yLocal);
        }
    }

    // Contract every row to its first 64 FP32 lanes across all rows at once,
    // then emit one compact FP32 sum per row. This path requires H to be a
    // multiple of 64 and keeps the reduce result vector-consumable.
    __aicore__ inline void ReduceAlignedRows(const AscendC::LocalTensor<float>& dst,
                                             const AscendC::LocalTensor<float>& src,
                                             int32_t rows, int32_t H) {
        constexpr int32_t oneRepeatFloat = 64;
        const uint8_t repStride = static_cast<uint8_t>(H / 8);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float>(0, rows * oneRepeatFloat);
        AscendC::BinaryRepeatParams binaryParams(1, 1, 1, repStride,
                                                  repStride, repStride);
        for (int32_t i = 1; i < H / oneRepeatFloat; ++i) {
            const int32_t offset = i * oneRepeatFloat;
            AscendC::Add<float, false>(src, src, src[offset],
                                       AscendC::MASK_PLACEHOLDER, 1,
                                       binaryParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetVectorMask<float>(0, rows * oneRepeatFloat);
        AscendC::RepeatReduceSum<float, false>(
            dst, src, 1, AscendC::MASK_PLACEHOLDER, 1, 1, 1, repStride);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Expand compact row scales to [rows, H] using the hardware broadcast
    // intrinsic. rows must be a multiple of 8 and H a multiple of 64.
    __aicore__ inline void BroadcastRowScales(const AscendC::LocalTensor<float>& dst,
                                               const AscendC::LocalTensor<float>& src,
                                               int32_t rows, int32_t H) {
        constexpr int32_t oneRepeatFloat = 64;
        const uint8_t repStride = static_cast<uint8_t>(H / 8);
        AscendC::BrcbRepeatParams brcParams(static_cast<uint16_t>(H / 8),
                                             static_cast<uint16_t>(H));
        AscendC::Brcb<float>(dst, src, static_cast<uint8_t>(rows / 8), brcParams);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float>(0, rows * oneRepeatFloat);
        AscendC::UnaryRepeatParams unaryParams(1, 0, repStride, repStride);
        for (int32_t i = 0; i < H / oneRepeatFloat; ++i) {
            const int32_t offset = i * oneRepeatFloat;
            AscendC::Adds<float, false>(dst[offset], dst, 0,
                                         AscendC::MASK_PLACEHOLDER, 1,
                                         unaryParams);
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Multiply [rows, H] by the shared [H] weight vector without leaving UB.
    __aicore__ inline void MultiplyWeightRows(const AscendC::LocalTensor<float>& srcDst,
                                              const AscendC::LocalTensor<float>& weight,
                                              int32_t rows, int32_t H) {
        constexpr int32_t oneRepeatFloat = 64;
        const uint8_t repStride = static_cast<uint8_t>(H / 8);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float>(0, rows * oneRepeatFloat);
        AscendC::BinaryRepeatParams binaryParams(1, 1, 1, repStride,
                                                  repStride, 0);
        for (int32_t i = 0; i < H / oneRepeatFloat; ++i) {
            const int32_t offset = i * oneRepeatFloat;
            AscendC::Mul<float, false>(srcDst[offset], srcDst[offset],
                                       weight[offset],
                                       AscendC::MASK_PLACEHOLDER, 1,
                                       binaryParams);
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

    __aicore__ inline void ProcessWholeRows() {
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<float> resoFp32 = resoFp32Buf.Get<float>();
        AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf.Get<float>();

        // Weight loaded once (full row, zero-padded to alignH), reused per row.
        LoadWeightRow(weightFp32, 0, this->hiddenSize, this->alignedHidden);

        const int32_t H = this->hiddenSize;
        const int32_t alignH = this->alignedHidden;
        const float invH = 1.0f / static_cast<float>(H);

        for (int64_t row = this->startRow; row < this->endRow; ++row) {
            uint64_t base = static_cast<uint64_t>(row) * static_cast<uint64_t>(H);

            // --- Load x, residual (FP16 GM -> UB) ---
            AscendC::LocalTensor<half> xLocal = inQueX.AllocTensor<half>();
            AscendC::LocalTensor<half> resLocal = inQueRes.AllocTensor<half>();
            CopyInRow(xLocal, xGm, base);
            CopyInRow(resLocal, residualGm, base);
            inQueX.EnQue(xLocal);
            inQueRes.EnQue(resLocal);
            xLocal = inQueX.DeQue<half>();
            resLocal = inQueRes.DeQue<half>();

            // residual_out (FP32) = Cast(x) + Cast(residual)
            AscendC::Cast(resoFp32, xLocal, AscendC::RoundMode::CAST_NONE, alignH);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(sq, resLocal, AscendC::RoundMode::CAST_NONE, alignH);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(resoFp32, resoFp32, sq, alignH);
            AscendC::PipeBarrier<PIPE_V>();
            inQueX.FreeTensor(xLocal);
            inQueRes.FreeTensor(resLocal);

            // --- Write residual_out (FP32 -> FP16 GM) ---
            AscendC::LocalTensor<half> resOutLocal = outQueResOut.AllocTensor<half>();
            AscendC::Cast(resOutLocal, resoFp32, AscendC::RoundMode::CAST_NONE, alignH);
            AscendC::PipeBarrier<PIPE_V>();
            outQueResOut.EnQue(resOutLocal);
            resOutLocal = outQueResOut.DeQue<half>();
            CopyOutRow(resOutLocal, residualOutGm, base);
            outQueResOut.FreeTensor(resOutLocal);

            // --- Reduce sum(residual_out^2) over the row (FP32) ---
            AscendC::Mul(sq, resoFp32, resoFp32, alignH);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceNormal(scalar, sq, H);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            float sumSq = scalar.GetValue(0);

            // rstd = 1 / sqrt(meanSq + eps). The 910B Rsqrt intrinsic is a fast
            // approximation (~0.2% error, enough to blow the fp16 tol); instead
            // compute rms = Sqrt(mean+eps) and divide, matching the fp32 golden
            // (torch.sqrt then divide) closely. Sqrt+Div are both ~1-ULP.
            float meanPlusEps = sumSq * invH + this->eps;
            AscendC::Duplicate<float>(sq, meanPlusEps, alignH);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt<float>(sq, sq, alignH);              // sq = rms
            AscendC::PipeBarrier<PIPE_V>();

            // --- y = (residual_out / rms) * weight (FP32), cast FP16, write GM ---
            AscendC::Div(resoFp32, resoFp32, sq, alignH);        // /= rms
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(resoFp32, resoFp32, weightFp32, alignH);  // *= weight
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> yLocal = outQueY.AllocTensor<half>();
            AscendC::Cast(yLocal, resoFp32, AscendC::RoundMode::CAST_NONE, alignH);
            AscendC::PipeBarrier<PIPE_V>();
            outQueY.EnQue(yLocal);
            yLocal = outQueY.DeQue<half>();
            CopyOutRow(yLocal, yGm, base);
            outQueY.FreeTensor(yLocal);
        }
    }

    // ------------------------------------------------------------------
    //  Chunked-row path (H > UB tile): two streaming passes per row.
    //  Pass 1: stream chunks, accumulate sum(residual_out^2) -> rstd.
    //  Pass 2: stream chunks, apply rstd*weight, write y + residual_out.
    //  (Weight is streamed per chunk in pass 2.)
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessChunkedRows() {
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<float> resoFp32 = resoFp32Buf.Get<float>();
        AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf.Get<float>();

        const int32_t H = this->hiddenSize;
        const int32_t chunkElems = this->tileElems;   // multiple of ALIGN_NUM
        const float invH = 1.0f / static_cast<float>(H);

        for (int64_t row = this->startRow; row < this->endRow; ++row) {
            uint64_t base = static_cast<uint64_t>(row) * static_cast<uint64_t>(H);

            // --- Pass 1: residual_out + accumulate sum-of-squares ---
            float sumSq = 0.0f;
            int32_t off = 0;
            while (off < H) {
                int32_t n = (H - off > chunkElems) ? chunkElems : (H - off);
                int32_t nAlign = (n + this->alignNum - 1) / this->alignNum * this->alignNum;

                AscendC::LocalTensor<half> xLocal = inQueX.AllocTensor<half>();
                AscendC::LocalTensor<half> resLocal = inQueRes.AllocTensor<half>();
                CopyInChunk(xLocal, xGm, base + off, n, nAlign);
                CopyInChunk(resLocal, residualGm, base + off, n, nAlign);
                inQueX.EnQue(xLocal);
                inQueRes.EnQue(resLocal);
                xLocal = inQueX.DeQue<half>();
                resLocal = inQueRes.DeQue<half>();

                AscendC::Cast(resoFp32, xLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Cast(sq, resLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(resoFp32, resoFp32, sq, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(sq, resoFp32, resoFp32, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                ReduceNormal(scalar, sq, n);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                sumSq += scalar.GetValue(0);

                inQueX.FreeTensor(xLocal);
                inQueRes.FreeTensor(resLocal);
                off += n;
            }

            float meanPlusEps = sumSq * invH + this->eps;

            // --- Pass 2: recompute residual_out, apply rstd*weight, write y + res_out ---
            off = 0;
            while (off < H) {
                int32_t n = (H - off > chunkElems) ? chunkElems : (H - off);
                int32_t nAlign = (n + this->alignNum - 1) / this->alignNum * this->alignNum;

                AscendC::LocalTensor<half> xLocal = inQueX.AllocTensor<half>();
                AscendC::LocalTensor<half> resLocal = inQueRes.AllocTensor<half>();
                CopyInChunk(xLocal, xGm, base + off, n, nAlign);
                CopyInChunk(resLocal, residualGm, base + off, n, nAlign);
                inQueX.EnQue(xLocal);
                inQueRes.EnQue(resLocal);
                xLocal = inQueX.DeQue<half>();
                resLocal = inQueRes.DeQue<half>();

                AscendC::Cast(resoFp32, xLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Cast(sq, resLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(resoFp32, resoFp32, sq, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                inQueX.FreeTensor(xLocal);
                inQueRes.FreeTensor(resLocal);

                // residual_out -> GM (FP16)
                AscendC::LocalTensor<half> resOutLocal = outQueResOut.AllocTensor<half>();
                AscendC::Cast(resOutLocal, resoFp32, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                outQueResOut.EnQue(resOutLocal);
                resOutLocal = outQueResOut.DeQue<half>();
                CopyOutChunk(resOutLocal, residualOutGm, base + off, n);
                outQueResOut.FreeTensor(resOutLocal);

                // y = (residual_out / rms) * weight  (Div+Sqrt path, see whole-row).
                AscendC::Duplicate<float>(sq, meanPlusEps, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sqrt<float>(sq, sq, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(resoFp32, resoFp32, sq, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                LoadWeightRow(weightFp32, off, n, nAlign);  // weight chunk for this offset
                AscendC::Mul(resoFp32, resoFp32, weightFp32, nAlign);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::LocalTensor<half> yLocal = outQueY.AllocTensor<half>();
                AscendC::Cast(yLocal, resoFp32, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                outQueY.EnQue(yLocal);
                yLocal = outQueY.DeQue<half>();
                CopyOutChunk(yLocal, yGm, base + off, n);
                outQueY.FreeTensor(yLocal);

                off += n;
            }
        }
    }

    // ------------------------------------------------------------------
    //  Helpers
    // ------------------------------------------------------------------
    // Load `realN` weight elements from GM offset `off`, zero-pad the tail up
    // to `nAlign` (nAlign >= realN, both multiples of ALIGN_NUM), and Cast them
    // into the FP32 weight tile `wFp32[0..nAlign)`. The padded tail is zero, so
    // the later Mul(reso, reso, wFp32, nAlign) is correct for the real elements
    // and harmless (×0) for the tail — which is never written back anyway
    // (CopyOut uses blockLen = n bytes).
    __aicore__ inline void LoadWeightRow(AscendC::LocalTensor<float>& wFp32,
                                         int32_t off, int32_t realN, int32_t nAlign) {
        AscendC::LocalTensor<half> wHalf = weightHalfBuf.Get<half>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(realN * sizeof(half));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<half> padParams;
        padParams.isPad = (realN < nAlign);
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint16_t>(nAlign - realN);
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(wHalf, weightGm[static_cast<uint64_t>(off)], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(wFp32, wHalf, AscendC::RoundMode::CAST_NONE, nAlign);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Copy a full aligned row (alignedHidden elems) from GM half -> UB half,
    // with zero-padding of the tail when hiddenSize < alignedHidden.
    __aicore__ inline void CopyInRow(AscendC::LocalTensor<half>& dst,
                                     AscendC::GlobalTensor<half>& src, uint64_t off) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(this->hiddenSize * sizeof(half));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<half> padParams;
        padParams.isPad = (this->hiddenSize < this->alignedHidden);
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint16_t>(this->alignedHidden - this->hiddenSize);
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(dst, src[off], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Copy `n` elems (padded to nAlign) from GM half -> UB half.
    __aicore__ inline void CopyInChunk(AscendC::LocalTensor<half>& dst,
                                       AscendC::GlobalTensor<half>& src,
                                       uint64_t off, int32_t n, int32_t nAlign) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(n * sizeof(half));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<half> padParams;
        padParams.isPad = (n < nAlign);
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint16_t>(nAlign - n);
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(dst, src[off], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Copy a full row (hiddenSize elems) from UB half -> GM half. Only the first
    // hiddenSize elements are written (blockLen = hiddenSize * sizeof(half) bytes).
    __aicore__ inline void CopyOutRow(AscendC::LocalTensor<half>& src,
                                      AscendC::GlobalTensor<half>& dst, uint64_t off) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(this->hiddenSize * sizeof(half));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst[off], src, copyParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Copy `n` elems from UB half -> GM half (byte-granular blockLen).
    __aicore__ inline void CopyOutChunk(AscendC::LocalTensor<half>& src,
                                        AscendC::GlobalTensor<half>& dst,
                                        uint64_t off, int32_t n) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(n * sizeof(half));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst[off], src, copyParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // High-performance FP32 reduce: BlockReduceSum loop + WholeReduceSum.
    // Reduces the first `totalElements` of src into dst[0] (one FP32 scalar).
    // Uses SetMaskCount + SetVectorMask<COUNTER>(totalElements) so only the
    // first totalElements participate (the UB tail, if any, is ignored).
    // NOTE: this clobbers `src` in place (BlockReduceSum writes partial sums
    // back into it); callers must not rely on src afterwards.
    __aicore__ inline void ReduceNormal(const AscendC::LocalTensor<float>& dst,
                                        const AscendC::LocalTensor<float>& src,
                                        const int totalElements) {
        constexpr int elemsPerBlock = 32 / sizeof(float);   // 8
        int currentLen = totalElements;
        AscendC::SetMaskCount();
        while (currentLen > (elemsPerBlock * 8)) {
            int blockCount = (currentLen + elemsPerBlock - 1) / elemsPerBlock;
            int repeat = (blockCount + 7) / 8;
            AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(currentLen);
            AscendC::BlockReduceSum<float, false>(src, src, repeat,
                                                 AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(currentLen);
        AscendC::WholeReduceSum<float, false>(dst, src, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

private:
    AscendC::TPipe* pipe;
    int32_t blockIdx;
    int32_t batchSize;
    int32_t hiddenSize;
    int32_t alignedHidden;
    int32_t alignNum;
    int32_t tileElems;
    int32_t rowTile;
    int64_t startRow;
    int64_t endRow;
    float eps;

    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> residualGm;
    AscendC::GlobalTensor<half> weightGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> residualOutGm;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueRes;
    AscendC::TQue<AscendC::TPosition::VECOUT, OUT_BUFFER_NUM> outQueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, OUT_BUFFER_NUM> outQueResOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightHalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightFp32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resoFp32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sqBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
};


extern "C" __global__ __aicore__ void fused_add_rms_norm(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                                                          GM_ADDR y, GM_ADDR residual_out,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    KernelFusedAddRmsNorm op;
    op.Init(x, residual, weight, y, residual_out, tilingData, &pipe);
    op.Process();
}
