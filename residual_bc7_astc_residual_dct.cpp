namespace xbc7
{
	static const int g_baseline_jpeg_y[8][8] =
	{
		// DC element modified (from 16) so bilinear fetches near (0,0) grab a smaller quant table value, protecting most important LF coefficients
		{  4, 11, 10, 16, 24, 40, 51, 61 },
		{ 12, 12, 14, 19, 26, 58, 60, 55 },
		{ 14, 13, 16, 24, 40, 57, 69, 56 },
		{ 14, 17, 22, 29, 51, 87, 80, 62 },
		{ 18, 22, 37, 56, 68,109,103, 77 },
		{ 24, 35, 55, 64, 81,104,113, 92 },
		{ 49, 64, 78, 87,103,121,120,101 },
		{ 72, 92, 95, 98,112,100,103, 99 }
	};

	// centers at (0,0)
	static inline float sample_jpeg_quant(const int Q8[8][8], float i, float j)
	{
		i = basisu::minimum(basisu::maximum(i, 0.0f), 7.0f);
		j = basisu::minimum(basisu::maximum(j, 0.0f), 7.0f);
		int i0 = (int)floorf(i), j0 = (int)floorf(j);
		int i1 = basisu::minimum(i0 + 1, 7), j1 = basisu::minimum(j0 + 1, 7);
		float ti = i - i0, tj = j - j0;
		float a = (1 - ti) * Q8[j0][i0] + ti * Q8[j0][i1];
		float b = (1 - ti) * Q8[j1][i0] + ti * Q8[j1][i1];
		return (1 - tj) * a + tj * b;
	}

	static const uint8_t g_zigzag4x4_xy[16][2] = // [index][X,Y]
	{
		{ 0, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 2 },
		{ 1, 1 },
		{ 2, 0 },
		{ 3, 0 },
		{ 2, 1 },
		{ 1, 2 },
		{ 0, 3 },
		{ 1, 3 },
		{ 2, 2 },
		{ 3, 1 },
		{ 3, 2 },
		{ 2, 3 },
		{ 3, 3 }
	};

	void compute_quant_table(float q,
		uint32_t grid_width, uint32_t grid_height,
		float level_scale, int* dct_quant_tab,
		int block_width = 4, int block_height = 4)
	{
		assert(q > 0.0f);

		dct_quant_tab[0] = 1;

		if (q >= 100.0f)
		{
			for (uint32_t y = 0; y < grid_height; y++)
				for (uint32_t x = 0; x < grid_width; x++)
					if (x || y)
						dct_quant_tab[x + y * grid_width] = 1;
			return;
		}
				
		const int Bx = block_width, By = block_height;

		const float sx = (float)8.0f / (float)Bx;
		const float sy = (float)8.0f / (float)By;

		for (uint32_t y = 0; y < grid_height; y++)
		{
			float ny = float(y);
			float ry = ny * sy;

			for (uint32_t x = y ? 0 : 1; x < grid_width; x++)
			{
				int quant_scale = 0;

				assert(x || y);

				float nx = float(x);
				float rx = nx * sx;

				// sample from the JPEG baseline luma 8x8 DCT quant matrix
				// this is an approximation (we could do an actual desired radians per spatial sample search vs. each of the 8x8 basis vectors to find the best, most conservative mapping), 
				// but for 4x4 and 6x6 block sizes it's reasonable enough and simple/fast
				// at 4x4, the lowest frequencies are slightly more heavily quantized than we would want (but the quant table entries near DC are so similar it's doubtful it matters much if at all)
				float base = sample_jpeg_quant(g_baseline_jpeg_y, rx, ry);
								
#if 1
				if ((x + y) == 1)
					base *= .25f;
				else if ((x == 1) && (y == 1))
					base *= .75f;
#endif

				//quant_scale = (int)std::floor(base * level_scale + 0.5f);
				quant_scale = (int)(base * level_scale + 0.5f);
				assert(quant_scale == (int)std::floor(base * level_scale + 0.5f));

				quant_scale = basisu::maximum<int>(1, quant_scale);

#if 1
				if ((x + y) == 1)
				{
					const int MAX_QUANT_SCALE_AC_1_1 = 73; // 73
					quant_scale = minimum(quant_scale, MAX_QUANT_SCALE_AC_1_1);
				}
#endif

				dct_quant_tab[x + y * grid_width] = quant_scale;
			} // x
		} // y

		for (uint32_t y = 0; y < grid_height; y++)
		{
			for (uint32_t x = y + 1; x < grid_width; x++)
			{
				assert(x != y);

				const int a = dct_quant_tab[x + y * grid_width];
				const int b = dct_quant_tab[y + x * grid_width];

				const int c = maximum(a, b);

				dct_quant_tab[x + y * grid_width] = c;
				dct_quant_tab[y + x * grid_width] = c;
			}
		}
	}

	struct coeff
	{
		int16_t m_num_zeros; // number of zero AC coefficients before this one
		int16_t m_coeff; // both sign and mag, [-256,256], or INT16_MAX if last

		void clear()
		{
			m_num_zeros = 0;
			m_coeff = 0;
		}
	};

	typedef basisu::vector<coeff> coeff_vec;

	struct dct_syms
	{
		int16_t m_dc;		// [-256,256]
				
		coeff_vec m_ac_vals;

		void clear()
		{
			m_dc = 0;
			m_ac_vals.resize(0);
		}
	};

	static const float DEADZONE_ALPHA = 0.5f;

	static const float g_scale_quant_steps[3] =
	{
		1.35588217f, // 4 (2-bits)
		1.24573100f, // 8 (3-bits)
		1.15431654f, // 16 (4-bits)
	};

	static inline uint32_t get_weight_size_index_from_bits(uint32_t num_weight_bits)
	{
		switch (num_weight_bits)
		{
		case 2: return 0;
		case 3: return 1;
		case 4: return 2;
		default:
			assert(0);
			return 0;
		}
	}

	class xbc7_weight_grid_dct
	{
	public:
		xbc7_weight_grid_dct()
		{
		}

		void init()
		{
			m_dct.init(BLOCK_HEIGHT, BLOCK_WIDTH);
		}
				
		void forward(
			float global_q, uint32_t plane_index,
			const int *pWeight_predictions, // may be nullptr
			const basist::bc7u::log_bc7_block& log_blk, 
			dct_syms &syms,
			basist::astc_ldr_t::fvec &dct_work)
		{
			syms.clear();

			float orig_weights[16];
			for (uint32_t i = 0; i < 16; i++)
			{
				const int predicted_weight = pWeight_predictions ? pWeight_predictions[i] : 0;
				assert((predicted_weight >= 0) && (predicted_weight <= 64));

				orig_weights[i] = (float)(basist::bc7u::dequant_weight(log_blk.m_weights[plane_index][i], log_blk.m_weight_bits[plane_index]) - predicted_weight);
			}
						
			float dct_weights[16];

			m_dct.forward(orig_weights, dct_weights, dct_work);

			const float span_len = get_max_span_len(log_blk, plane_index);
			const float level_scale = compute_level_scale(global_q, span_len, log_blk.m_weight_bits[plane_index]);

			int dct_quant_tab[16];
			compute_quant_table(global_q, 4, 4, level_scale, dct_quant_tab, 4, 4);

			int dct_coeffs[16];

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					if (!x && !y)
					{
						dct_coeffs[0] = clamp<int>((int)std::round(dct_weights[0]), -256, 256);
						continue;
					}

					const int levels = dct_quant_tab[x + y * 4];

					float d = dct_weights[x + y * 4];

					int id = quantize_deadzone(d, levels, DEADZONE_ALPHA, x, y);

					dct_coeffs[x + y * 4] = clamp<int>(id, -256, 256);

				} // x

			}  // y

			syms.m_dc = safe_cast_int16(dct_coeffs[0]);

			syms.m_ac_vals.reserve(17);

			int total_zeros = 0;
			for (uint32_t i = 1; i < 16; i++)
			{
				const uint32_t dct_idx = g_zigzag4x4_xy[i][0] + (g_zigzag4x4_xy[i][1] * 4);
				assert(dct_idx);

				int ac_coeff = dct_coeffs[dct_idx];
				if (!ac_coeff)
				{
					total_zeros++;
					continue;
				}

				coeff cf;
				cf.m_num_zeros = basisu::safe_cast_int16(total_zeros);
				cf.m_coeff = basisu::safe_cast_int16(ac_coeff);

				syms.m_ac_vals.push_back(cf);

				total_zeros = 0;
			}

			if (total_zeros)
			{
				coeff cf;
				cf.m_num_zeros = basisu::safe_cast_int16(total_zeros);
				cf.m_coeff = INT16_MAX;
				syms.m_ac_vals.push_back(cf);
			}
		}

		bool inverse(
			float global_q, uint32_t plane_index,
			const int* pWeight_predictions, // may be nullptr
			const dct_syms& syms,
			basist::bc7u::log_bc7_block& log_blk,
			basist::astc_ldr_t::fvec& dct_work)
		{
			const float span_len = get_max_span_len(log_blk, plane_index);

			const float level_scale = compute_level_scale(global_q, span_len, log_blk.m_weight_bits[plane_index]);

			int dct_quant_tab[16];
			compute_quant_table(global_q, 4, 4, level_scale, dct_quant_tab, 4, 4);

			float dct_weights[16];
			basisu::clear_obj(dct_weights);

			dct_weights[0] = (float)syms.m_dc;

			uint32_t zig_idx = 1;
			uint32_t coeff_ofs = 0;
			while (coeff_ofs < syms.m_ac_vals.size())
			{
				const uint32_t run_len = syms.m_ac_vals[coeff_ofs].m_num_zeros;
				const int coeff = syms.m_ac_vals[coeff_ofs].m_coeff;
				coeff_ofs++;

				if ((run_len + zig_idx) > 16)
					return false;

				zig_idx += run_len;

				if (zig_idx >= 16)
					break;

				assert(coeff != INT16_MAX);

				const int x = g_zigzag4x4_xy[zig_idx][0];
				const int y = g_zigzag4x4_xy[zig_idx][1];
				const int dct_idx = x + (y * 4);

				const int quant = dct_quant_tab[dct_idx];

				dct_weights[dct_idx] = dequant_deadzone(coeff, quant, DEADZONE_ALPHA, x, y);

				zig_idx++;
			}

			float idct_weights[astc_helpers::MAX_BLOCK_PIXELS];

			m_dct.inverse(dct_weights, idct_weights, dct_work);

			for (uint32_t i = 0; i < 16; i++)
			{
				log_blk.m_weights[plane_index][i] = basist::bc7u::quant_weight(
					basisu::clamp<int>(fast_roundf_int(idct_weights[i] + (pWeight_predictions ? pWeight_predictions[i] : 0)), 0, 64),
					log_blk.m_weight_bits[plane_index]);
			}

			return true;
		}
				
	private:
		static const uint32_t BLOCK_WIDTH = 4;
		static const uint32_t BLOCK_HEIGHT = 4;
		
		basist::astc_ldr_t::dct2f m_dct;
				
		// Adaptive quantization
		float compute_level_scale(float q, float span_len, uint32_t num_weight_bits)
		{
			const uint32_t weight_size_index = get_weight_size_index_from_bits(num_weight_bits);
			
			// Standard JPEG quality factor calcs
			// TODO: Precompute this once
			float level_scale;
			q = basisu::clamp(q, 1.0f, 100.0f);
			if (q < 50.0f)
				level_scale = 5000.0f / q;
			else
				level_scale = 200.0f - 2.0f * q;

			level_scale *= (1.0f / 100.0f); // because JPEG's quant table is scaled by 100

			//const float span_floor = 28.0f;
			const float span_floor = 14.0f;
			//const float adaptive_factor = 255.0f / maximum<float>(span_len, span_floor);
			// 64.0 = dynamic range adjustment (JPEG uses 255)
			// divide by span len to adjust adaptive low/high values per-block (JPEG always uses effective span=0-255)
			// actually (64/255) * 255/max(span_len, span_floor)
			float adaptive_factor = 64.0f / basisu::maximum<float>(span_len, span_floor);

			// input signal scalar quantization noise will be distributed between multiple AC coefficients - compensate by adaptively adjusting the quant step size
			float weight_quant_adaptive_factor = g_scale_quant_steps[weight_size_index];
			adaptive_factor *= weight_quant_adaptive_factor;

			// (Adaptive quant)
			level_scale *= adaptive_factor;

			// The higher the level_scale, the more quantized DCT coefficients will be and vice versa.

			return level_scale;
		}

		// Needed by AQ
		float get_max_span_len(const basist::bc7u::log_bc7_block& log_blk, uint32_t plane_index) const
		{
			float span_len = 0.0f;

			if (log_blk.is_dual_plane())
			{
				basist::color_rgba ep[2];

				basist::bc7u::unpack_endpoints(log_blk, ep, 0);

				const basist::color_rgba& l = ep[0];
				const basist::color_rgba& h = ep[1];

				for (uint32_t c = 0; c < 4; c++)
				{
					// get the weight plane used by this endpoint channel (NOT the decoded pixel channel, which is after any mode 4/5 channel swapping/rotation)
					const uint32_t endpoint_chan_plane = log_blk.get_endpoint_channel_weight_plane(c);

					if (endpoint_chan_plane == plane_index)
					{
						span_len += basisu::squaref((float)h[c] - (float)l[c]);
					}
				}

				span_len = sqrtf(span_len);
			}
			else
			{
				assert(!plane_index);

				for (uint32_t i = 0; i < log_blk.m_num_partitions; i++)
				{
					basist::color_rgba ep[2];

					basist::bc7u::unpack_endpoints(log_blk, ep, i);

					const basist::color_rgba& l = ep[0];
					const basist::color_rgba& h = ep[1];

					float part_span_len = sqrtf(
						basisu::squaref((float)h.r - (float)l.r) + basisu::squaref((float)h.g - (float)l.g) + basisu::squaref((float)h.b - (float)l.b) + basisu::squaref((float)h.a - (float)l.a)
					);

					span_len = basisu::maximum(part_span_len, span_len);
				}
			}

			return span_len;
		}

		inline int quantize_deadzone(float d, int L, float alpha, uint32_t x, uint32_t y) const
		{
			assert((x < BLOCK_WIDTH) && (y < BLOCK_HEIGHT));

			if (((x == 1) && (y == 0)) ||
				((x == 0) && (y == 1)))
			{
				return (int)std::round(d / (float)L);
			}

			// L = quant step, alpha in [0,1.2] (typical 0.7–0.85)
			if (L <= 0)
				return 0;

			float s = fabsf(d);
			float tau = alpha * float(L);                 // half-width of the zero band

			if (s <= tau)
				return 0;                       // inside dead-zone towards zero

			// Quantize the residual outside the dead-zone with mid-tread rounding
			float qf = (s - tau) / float(L);
			int   q = (int)floorf(qf + 0.5f);            // ties-nearest
			return (d < 0.0f) ? -q : q;
		}

		inline float dequant_deadzone(int q, int L, float alpha, uint32_t x, uint32_t y) const
		{
			assert((x < BLOCK_WIDTH) && (y < BLOCK_HEIGHT));

			if (((x == 1) && (y == 0)) ||
				((x == 0) && (y == 1)))
			{
				return (float)q * (float)L;
			}

			if (q == 0 || L <= 0)
				return 0.0f;

			float tau = alpha * float(L);
			float mag = tau + float(abs(q)) * float(L);   // center of the (nonzero) bin
			return (q < 0) ? -mag : mag;
		}
	};
		
} // namespace xbc7

static inline uint32_t index_from_xy(uint32_t x, uint32_t y) { assert((x < 4) && (y < 4));  return x + y * 4; }

static bool bc7_test(int argc, const char *argv[])
{
	basisu::rand rnd;
	rnd.seed(1000);

	enable_debug_printf(true);

	if (argc != 2)
		return false;

	const char* pFilename = argv[1];

	image orig_img;
	if (!load_png(pFilename, orig_img))
		return false;

	const bool srgb_flag = true;

	const uint32_t block_width = 4;
	const uint32_t block_height = 4;
	const uint32_t total_block_pixels = block_width * block_height;

	const uint32_t width = orig_img.get_width();
	const uint32_t height = orig_img.get_height();
	const uint32_t num_blocks_x = (width + block_width - 1) / block_width;
	const uint32_t num_blocks_y = (height + block_height - 1) / block_height;
	const uint32_t total_blocks = num_blocks_x * num_blocks_y;
		
	vector2D<basist::bc7u::log_bc7_block> log_blks(num_blocks_x, num_blocks_y);

	for (uint32_t by = 0; by < num_blocks_y; by++)
	{
		for (uint32_t bx = 0; bx < num_blocks_x; bx++)
		{
			color_rgba orig_block[16];
			orig_img.extract_block_clamped(orig_block, bx * 4, by * 4, 4, 4);

			basist::bc7u::phys_bc7_block phys_blk;
			basist::bc7f::fast_pack_bc7_auto_rgba(phys_blk.m_bytes, (basist::color_rgba*)orig_block, basist::bc7f::cPackBC7FlagDefaultPartiallyAnalytical);

			basist::bc7u::log_bc7_block& log_blk = log_blks(bx, by);

			bool unpack_status = basist::bc7u::unpack_bc7(&phys_blk, log_blk);
			if (!unpack_status)
			{
				assert(0);
				return false;
			}
		}
	}

	image out_img(width, height);

	xbc7::xbc7_weight_grid_dct weight_grid_dct;
	weight_grid_dct.init();

	basist::astc_ldr_t::fvec dct_work;

	const float global_q = 75;// 9.0f;

	uint32_t total_ac_syms = 0;

	uint_vec cand_hist(17);

	vector2D<basist::bc7u::phys_bc7_block> phys_blocks(num_blocks_x, num_blocks_y);

	for (uint32_t by = 0; by < num_blocks_y; by++)
	{
		fmt_printf(".");

		for (uint32_t bx = 0; bx < num_blocks_x; bx++)
		{
			color_rgba orig_block[16];
			orig_img.extract_block_clamped(orig_block, bx * 4, by * 4, 4, 4);

			const basist::bc7u::log_bc7_block* pLeft_diag_log_blk = (bx && by) ? &log_blks(bx - 1, by - 1) : nullptr;
			const basist::bc7u::log_bc7_block* pRight_diag_log_blk = (((bx + 1) < num_blocks_x) && by) ? &log_blks(bx + 1, by - 1) : nullptr;
			const basist::bc7u::log_bc7_block* pUp_log_blk = by ? &log_blks(bx, by - 1) : nullptr;
			const basist::bc7u::log_bc7_block* pLeft_log_blk = bx ? &log_blks(bx - 1, by) : nullptr;
			
			basist::bc7u::log_bc7_block& log_blk = log_blks(bx, by);
			const basist::bc7u::log_bc7_block orig_log_blk(log_blk);

			basist::bc7u::log_bc7_block best_cand_log_blk(orig_log_blk);
			uint64_t best_err = UINT64_MAX;
			uint32_t best_num_ac_syms = UINT32_MAX;
			
			if (!basist::bc7u::is_solid_blk(log_blk))
			{
				uint32_t best_cand_index = 0;

				const uint32_t TOTAL_CANDIDATES = 17;

				for (uint32_t cand_index = 0; cand_index < TOTAL_CANDIDATES; cand_index++)
				{
					const basist::bc7u::log_bc7_block* pCand_log_blk = nullptr;

					if (cand_index == 0)
					{

					}
					else
					{
						if (cand_index == 1)
							pCand_log_blk = pLeft_log_blk;
						else if (cand_index == 2)
							pCand_log_blk = pUp_log_blk;
						else if (cand_index == 3)
							pCand_log_blk = pLeft_diag_log_blk;
						else if (cand_index == 4)
							pCand_log_blk = pRight_diag_log_blk;
						else if (cand_index == 5)
							pCand_log_blk = pLeft_log_blk; // left edge
						else if (cand_index == 6)
							pCand_log_blk = pUp_log_blk; // upper edge
						else if (cand_index == 7)
							pCand_log_blk = (pLeft_log_blk && pUp_log_blk) ? pLeft_log_blk : nullptr; // left+upper edge blend
						else if (cand_index == 8) 
							pCand_log_blk = pLeft_log_blk; // reflect left
						else if (cand_index == 9)
							pCand_log_blk = pUp_log_blk; // reflect upper
						else if (cand_index == 10)
							pCand_log_blk = (pLeft_log_blk && pUp_log_blk) ? pLeft_log_blk : nullptr; // left+upper edge avg
						else if (cand_index == 11)
							pCand_log_blk = (pLeft_log_blk && pUp_log_blk) ? pLeft_log_blk : nullptr; // left+upper edge stronger distance blend
						else if (cand_index == 12)
							pCand_log_blk = (pLeft_log_blk && pUp_log_blk && pLeft_diag_log_blk) ? pLeft_log_blk : nullptr; // gradient
						else if (cand_index == 13)
							pCand_log_blk = (pLeft_log_blk && pUp_log_blk && pLeft_diag_log_blk) ? pLeft_log_blk : nullptr; // damped gradient
						else if (cand_index == 14)
							pCand_log_blk = (pLeft_diag_log_blk && pRight_diag_log_blk) ? pLeft_diag_log_blk : nullptr; // left/right diagonal avg
						else if (cand_index == 15)
							pCand_log_blk = (pLeft_diag_log_blk && pRight_diag_log_blk) ? pLeft_diag_log_blk : nullptr; // diagonal edge blend
						else if (cand_index == 16)
							pCand_log_blk = (pUp_log_blk && pLeft_diag_log_blk && pRight_diag_log_blk) ? pLeft_diag_log_blk : nullptr; // upper + diagonal edge blend

						if (!pCand_log_blk)
							continue;
					}

					basist::bc7u::log_bc7_block cand_log_blk(log_blk);

					uint32_t cand_total_ac_syms = 0;

					for (uint32_t p = 0; p < cand_log_blk.m_num_planes; p++)
					{
						const int* pWeight_predictions = nullptr;

						int weight_preds[16];
						if (pCand_log_blk)
						{
							for (uint32_t w = 0; w < 16; w++)
							{
								if (pCand_log_blk->is_dual_plane())
									weight_preds[w] = basist::bc7u::dequant_weight(pCand_log_blk->m_weights[p][w], pCand_log_blk->m_weight_bits[p]);
								else
									weight_preds[w] = basist::bc7u::dequant_weight(pCand_log_blk->m_weights[0][w], pCand_log_blk->m_weight_bits[0]);
							}

							int orig_weight_preds[16];
							memcpy(orig_weight_preds, weight_preds, sizeof(orig_weight_preds));

							if (cand_index == 5)
							{
								// left edge
								for (uint32_t y = 0; y < 4; y++)
									for (uint32_t x = 0; x < 4; x++)
										weight_preds[index_from_xy(x, y)] = orig_weight_preds[index_from_xy(3, y)];
							}
							else if (cand_index == 6)
							{
								// upper edge
								for (uint32_t y = 0; y < 4; y++)
									for (uint32_t x = 0; x < 4; x++)
										weight_preds[index_from_xy(x, y)] = orig_weight_preds[index_from_xy(x, 3)];
							}
							else if ((cand_index == 7) || (cand_index == 10) || (cand_index == 11))
							{
								// left+upper edge blend variants.
								// pCand_log_blk is pLeft_log_blk here, so orig_weight_preds contains the left block.
								// Pull upper edge directly from pUp_log_blk.

								int upper_edge[4];

								for (uint32_t x = 0; x < 4; x++)
								{
									const uint32_t w = index_from_xy(x, 3); // upper block's bottom edge

									if (pUp_log_blk->is_dual_plane())
										upper_edge[x] = basist::bc7u::dequant_weight(pUp_log_blk->m_weights[p][w], pUp_log_blk->m_weight_bits[p]);
									else
										upper_edge[x] = basist::bc7u::dequant_weight(pUp_log_blk->m_weights[0][w], pUp_log_blk->m_weight_bits[0]);
								}

								for (uint32_t y = 0; y < 4; y++)
								{
									const int left_val = orig_weight_preds[index_from_xy(3, y)]; // left block's right edge

									for (uint32_t x = 0; x < 4; x++)
									{
										const int upper_val = upper_edge[x];
										int pred;

										if (cand_index == 7)
										{
											// Existing distance-weighted blend.
											const int wl = 4 - static_cast<int>(x); // 4,3,2,1
											const int wu = 4 - static_cast<int>(y); // 4,3,2,1
											const int den = wl + wu;

											pred = (wl * left_val + wu * upper_val + (den >> 1)) / den;
										}
										else if (cand_index == 10)
										{
											// Simple average.
											pred = (left_val + upper_val + 1) >> 1;
										}
										else // cand_index == 11
										{
											// Stronger distance weighting: trust the nearest edge more.
											const int dx = 4 - static_cast<int>(x); // 4,3,2,1
											const int dy = 4 - static_cast<int>(y); // 4,3,2,1
											const int wl = dx * dx; // 16,9,4,1
											const int wu = dy * dy; // 16,9,4,1
											const int den = wl + wu;

											pred = (wl * left_val + wu * upper_val + (den >> 1)) / den;
										}

										weight_preds[index_from_xy(x, y)] = pred;
									}
								}
							}
							else if (cand_index == 8)
							{
								// reflect left
								for (uint32_t y = 0; y < 4; y++)
									for (uint32_t x = 0; x < 4; x++)
										weight_preds[index_from_xy(x, y)] = orig_weight_preds[index_from_xy(3 - x, y)];
							}
							else if (cand_index == 9)
							{
								// reflect upper
								for (uint32_t y = 0; y < 4; y++)
									for (uint32_t x = 0; x < 4; x++)
										weight_preds[index_from_xy(x, y)] = orig_weight_preds[index_from_xy(x, 3 - y)];
							}
							else if ((cand_index == 12) || (cand_index == 13))
							{
								int upper_edge[4];

								for (uint32_t x = 0; x < 4; x++)
								{
									const uint32_t w = index_from_xy(x, 3); // upper block's bottom edge

									if (pUp_log_blk->is_dual_plane())
										upper_edge[x] = basist::bc7u::dequant_weight(pUp_log_blk->m_weights[p][w], pUp_log_blk->m_weight_bits[p]);
									else
										upper_edge[x] = basist::bc7u::dequant_weight(pUp_log_blk->m_weights[0][w], pUp_log_blk->m_weight_bits[0]);
								}

								const uint32_t corner_w = index_from_xy(3, 3); // upper-left block's bottom-right

								int corner_val;
								if (pLeft_diag_log_blk->is_dual_plane())
									corner_val = basist::bc7u::dequant_weight(pLeft_diag_log_blk->m_weights[p][corner_w], pLeft_diag_log_blk->m_weight_bits[p]);
								else
									corner_val = basist::bc7u::dequant_weight(pLeft_diag_log_blk->m_weights[0][corner_w], pLeft_diag_log_blk->m_weight_bits[0]);

								for (uint32_t y = 0; y < 4; y++)
								{
									const int left_val = orig_weight_preds[index_from_xy(3, y)]; // left block's right edge

									for (uint32_t x = 0; x < 4; x++)
									{
										const int upper_val = upper_edge[x];

										int grad = left_val + upper_val - corner_val;
										grad = basisu::clamp<int>(grad, 0, 64); // or your clamp helper

										if (cand_index == 12)
										{
											weight_preds[index_from_xy(x, y)] = grad;
										}
										else
										{
											// Damped gradient: blend gradient with your proven #7 predictor.
											const int wl = 4 - static_cast<int>(x);
											const int wu = 4 - static_cast<int>(y);
											const int den = wl + wu;
											const int blend7 = (wl * left_val + wu * upper_val + (den >> 1)) / den;

											weight_preds[index_from_xy(x, y)] = (grad + blend7 + 1) >> 1;
										}
									}
								}
							}
							else if (cand_index == 14)
							{
								// Average upper-left and upper-right diagonal blocks.
								// pCand_log_blk is pLeft_diag_log_blk here, so orig_weight_preds contains upper-left.
								// Pull upper-right directly from pRight_diag_log_blk.

								for (uint32_t w = 0; w < 16; w++)
								{
									int right_diag_val;

									if (pRight_diag_log_blk->is_dual_plane())
										right_diag_val = basist::bc7u::dequant_weight(
											pRight_diag_log_blk->m_weights[p][w],
											pRight_diag_log_blk->m_weight_bits[p]);
									else
										right_diag_val = basist::bc7u::dequant_weight(
											pRight_diag_log_blk->m_weights[0][w],
											pRight_diag_log_blk->m_weight_bits[0]);

									weight_preds[w] = (orig_weight_preds[w] + right_diag_val + 1) >> 1;
								}
							}
							else if (cand_index == 15)
							{
								// Blend upper-left block's right edge with upper-right block's left edge.
								// pCand_log_blk is pLeft_diag_log_blk, so orig_weight_preds contains upper-left.
								// Pull upper-right left edge directly from pRight_diag_log_blk.
								//
								// For each row y:
								//   L = upper-left[3,y]
								//   R = upper-right[0,y]
								// Then interpolate across x.

								int right_diag_left_edge[4];

								for (uint32_t y = 0; y < 4; y++)
								{
									const uint32_t w = index_from_xy(0, y); // upper-right block's left edge

									if (pRight_diag_log_blk->is_dual_plane())
										right_diag_left_edge[y] = basist::bc7u::dequant_weight(
											pRight_diag_log_blk->m_weights[p][w],
											pRight_diag_log_blk->m_weight_bits[p]);
									else
										right_diag_left_edge[y] = basist::bc7u::dequant_weight(
											pRight_diag_log_blk->m_weights[0][w],
											pRight_diag_log_blk->m_weight_bits[0]);
								}

								for (uint32_t y = 0; y < 4; y++)
								{
									const int left_val = orig_weight_preds[index_from_xy(3, y)]; // upper-left right edge
									const int right_val = right_diag_left_edge[y];                // upper-right left edge

									for (uint32_t x = 0; x < 4; x++)
									{
										// x=0 mostly left_val, x=3 mostly right_val.
										// Use 4-sample interpolation: 3/0, 2/1, 1/2, 0/3.
										const int pred = ((3 - static_cast<int>(x)) * left_val +
											static_cast<int>(x) * right_val + 1) / 3;

										weight_preds[index_from_xy(x, y)] = pred;
									}
								}
							}
							else if (cand_index == 16)
							{
								// Blend upper edge predictor with diagonal edge blend.
								//
								// upper_edge[x] = upper block's bottom edge
								// diag_blend[x,y] = horizontal interpolation between:
								//   upper-left block's right edge and upper-right block's left edge
								//
								// This combines direct top continuation with previous-row lateral structure.

								int upper_edge[4];
								int right_diag_left_edge[4];

								for (uint32_t x = 0; x < 4; x++)
								{
									const uint32_t up_w = index_from_xy(x, 3); // upper block's bottom edge

									if (pUp_log_blk->is_dual_plane())
										upper_edge[x] = basist::bc7u::dequant_weight(
											pUp_log_blk->m_weights[p][up_w],
											pUp_log_blk->m_weight_bits[p]);
									else
										upper_edge[x] = basist::bc7u::dequant_weight(
											pUp_log_blk->m_weights[0][up_w],
											pUp_log_blk->m_weight_bits[0]);
								}

								for (uint32_t y = 0; y < 4; y++)
								{
									const uint32_t rd_w = index_from_xy(0, y); // upper-right block's left edge

									if (pRight_diag_log_blk->is_dual_plane())
										right_diag_left_edge[y] = basist::bc7u::dequant_weight(
											pRight_diag_log_blk->m_weights[p][rd_w],
											pRight_diag_log_blk->m_weight_bits[p]);
									else
										right_diag_left_edge[y] = basist::bc7u::dequant_weight(
											pRight_diag_log_blk->m_weights[0][rd_w],
											pRight_diag_log_blk->m_weight_bits[0]);
								}

								for (uint32_t y = 0; y < 4; y++)
								{
									const int left_diag_right_val = orig_weight_preds[index_from_xy(3, y)]; // upper-left right edge
									const int right_diag_left_val = right_diag_left_edge[y];

									for (uint32_t x = 0; x < 4; x++)
									{
										// Same as #15: lateral predictor from upper-left/right diagonal edges.
										const int diag_blend =
											((3 - static_cast<int>(x)) * left_diag_right_val +
												static_cast<int>(x) * right_diag_left_val + 1) / 3;

										// Same as #6: direct upper edge replicated downward.
										const int up_val = upper_edge[x];

										// Trust upper edge more near y=0, trust diagonal lateral structure more lower in the block.
										const int wu = 4 - static_cast<int>(y); // 4,3,2,1
										const int wd = 1 + static_cast<int>(y); // 1,2,3,4
										const int den = wu + wd;                // always 5

										weight_preds[index_from_xy(x, y)] =
											(wu * up_val + wd * diag_blend + (den >> 1)) / den;
									}
								}
							}
							
							pWeight_predictions = weight_preds;
						}

						xbc7::dct_syms syms;

						weight_grid_dct.forward(global_q, p, pWeight_predictions, cand_log_blk, syms, dct_work);
												
						memset(cand_log_blk.m_weights[p], 0, 16);

						bool status = weight_grid_dct.inverse(global_q, p, pWeight_predictions, syms, cand_log_blk, dct_work);
						if (!status)
						{
							assert(0);
							return false;
						}

						cand_total_ac_syms += syms.m_ac_vals.size_u32();

					} // p

					if (cand_total_ac_syms < best_num_ac_syms)
					{
						best_cand_log_blk = cand_log_blk;
						best_cand_index = cand_index;
						best_num_ac_syms = cand_total_ac_syms;
					}

#if 0
					color_rgba cand_block_pixels[16];
					basist::bc7u::unpack_bc7(cand_log_blk, (basist::color_rgba*)cand_block_pixels);

					uint64_t cand_err = 0;
					for (uint32_t i = 0; i < 16; i++)
						cand_err += cand_block_pixels[i].get_dist2(orig_block[i]);

					if (cand_err < best_err)
					{
						best_err = cand_err;
						best_cand_log_blk = cand_log_blk;
						best_cand_index = cand_index;
					}
#endif
				}

				cand_hist[best_cand_index]++;
			}

			log_blk = best_cand_log_blk;
			total_ac_syms += best_num_ac_syms;

			color_rgba new_block_pixels[16];
			basist::bc7u::unpack_bc7(log_blk, (basist::color_rgba *)new_block_pixels);
			out_img.set_block_clipped(new_block_pixels, bx * 4, by * 4, 4, 4);

			bool pack_status = basist::bc7u::pack_bc7(log_blk, &phys_blocks(bx, by));
			assert(pack_status);
			if (!pack_status)
				return false;
						
		} // bx

	} // by

	save_png("out_img.png", out_img);

	fmt_printf("\nOK\n");

	fmt_printf("Total AC syms: {}, avg per block: {}\n", total_ac_syms, (float)total_ac_syms / (float)total_blocks);

	fmt_printf("Candidate histogram:\n");
	for (uint32_t i = 0; i < cand_hist.size(); i++)
		fmt_printf("{}: {}\n", i, cand_hist[i]);

	create_bc7_debug_images(
		width, height,
		phys_blocks.get_ptr(),
		"out_img");

	return false;
}
