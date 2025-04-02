#include "dataflow_transformer.h"

// ALL BAKED INTO 1 Large Function for now,
// but really should have subfunctions to do norms, attn, and mlp based on transformer block config...!

int submit_transformer_block(Dataflow_Handle * dataflow_handle, int compute_stream_id, void * X, Transformer_Block * transformer_block, Transformer_Block_Activations * activations) {

	int ret;


	DataflowDatatype fwd_dt = (transformer_block -> config).block_dt;
	DataflowDatatype compute_dt = (transformer_block -> config).compute_dt;

	int num_seqs = (activations -> config).num_seqs;
	int total_q = (activations -> config).total_q;
	int total_k = (activations -> config).total_k;

	
	int model_dim = (transformer_block -> config).model_dim;
	int kv_dim = (transformer_block -> config).kv_dim;
	int ffn_dim = (transformer_block -> config).ffn_dim;

	
	uint64_t workspaceBytes = (activations -> config).workspaceBytes;
	void * workspace = (activations -> config).workspace;

	// Assume weights are in col-major format.

	// But we want to process activations in row-major

	// Note that matmul interface assumes col-major storage format

	// Also note that FP8 tensor cores only available in TN format

	// During FWD pass we normally want:


	// Thus to compute Y = X @ W, 
	// we can do Y^T = W^T @ X^T
	// where from matmul perspective ^T means we interpret as row-major
	// However we store W as col-major so we need to transpose it.

	// Also for M, K, N (assuming X: (m, k), W (k, n))
	// we set M = n, K = k, N = m

	// The BWD pass is different because if we want dX's to be in row major we need:

	// dX = dY @ W^T
	// => dX^T = W @ dY^T

	// so if we store W in col-major format we shouldn't transpose it..

	// Now for bwd we set
	// M = k, K = n, N = m
	// where m, k, n are from fwd values of X, W, and Y.

	int to_transa = 1;
	int to_transb = 0;


	printf("Submitting Attention RMS Norm...!\n");

	ret = submit_rms_norm(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, model_dim, (transformer_block -> config).eps, 
						transformer_block -> w_attn_norm, X, activations -> x_temp, 
						activations -> attn_norm_weighted_sums, activations -> attn_norm_rms_vals);

	if (ret){
		fprintf(stderr, "Error: failed to submit attention norm...\n");
		return -1;
	}	



	printf("Submitting Q, K, V matmuls...!\n");

	// Q Proj
	ret = submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_q, activations -> x_temp, NULL, activations -> x_q,
					workspaceBytes, workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit q matmul proj...\n");
		return -1;
	}

	ret = submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, kv_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_k, activations -> x_temp, NULL, activations -> x_k_local,
					workspaceBytes, workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit k matmul proj...\n");
		return -1;
	}

	ret = submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, kv_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_v, activations -> x_temp, NULL, activations -> x_v_local,
					workspaceBytes, workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit k matmul proj...\n");
		return -1;
	}


	printf("Submitting RoPE...!\n");

	int num_q_heads = (transformer_block -> config).num_q_heads;
	int num_kv_heads = (transformer_block -> config).num_kv_heads;
	int head_dim = (transformer_block -> config).head_dim;


	uint64_t N = (uint64_t) total_q * (uint64_t) model_dim;

	ret = submit_rope(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						N, model_dim, head_dim, num_kv_heads, (transformer_block -> config).theta,
						(activations -> config).seq_positions, activations -> x_q, activations -> x_k_local);
	if (ret){
		fprintf(stderr, "Error: failed to submit rope...\n");
		return -1;
	}


	printf("Submitting Attention...!\n");

	// ensure workspace is zerod out beforehand....

	ret = (dataflow_handle -> set_mem)(dataflow_handle, compute_stream_id, workspace, 0, workspaceBytes);
	if (ret){
		fprintf(stderr, "Error: unable to set attention workspace mem to 0 before submitting...\n");
		return -1;
	}

	void * q_seq_offsets = (activations -> config).q_seq_offsets;
	void * q_seq_lens = (activations -> config).q_seq_lens;
	int max_seqlen_q = (activations -> config).max_seqlen_q;

	void * k_seq_offsets = (activations -> config).k_seq_offsets;
	void * k_seq_lens = (activations -> config).k_seq_lens;
	int max_seqlen_k = (activations -> config).max_seqlen_k;

	ret = submit_attention(dataflow_handle, compute_stream_id,
						 fwd_dt, 
						 num_seqs, total_q, total_k,
						 q_seq_offsets, q_seq_lens, max_seqlen_q,
						 k_seq_offsets, k_seq_lens, max_seqlen_k,
						 num_q_heads, num_kv_heads, head_dim,
						 activations -> x_q, activations -> x_k_local, activations -> x_v_local,
						 activations -> x_temp, activations -> softmax_lse, 
						 workspaceBytes, workspace);
	if (ret){
		fprintf(stderr, "Error: failed to submit attention...\n");
		return -1;
	}


	printf("Submitting Attention Output Matmul...!\n");


	ret = submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, fwd_dt, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q, 
					1.0, 1.0,
					transformer_block -> w_o, activations -> x_temp, X, activations -> x_o,
					workspaceBytes, workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit o matmul proj...\n");
		return -1;
	}


	printf("Submitting FFN RMS Norm...!\n");

	ret = submit_rms_norm(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, model_dim, (transformer_block -> config).eps, 
						transformer_block -> w_ffn_norm, activations -> x_o, activations -> x_temp, 
						activations -> ffn_norm_weighted_sums, activations -> ffn_norm_rms_vals);

	if (ret){
		fprintf(stderr, "Error: failed to submit ffn norm...\n");
		return -1;
	}


	printf("Submitting FFN w1 and w3 matmuls...!\n");

	ret = submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, ffn_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_1, activations -> x_temp, NULL, (activations -> x_1)[0],
					workspaceBytes, workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w1 matmul proj...\n");
		return -1;
	}

	ret = submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, ffn_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_3, activations -> x_temp, NULL, (activations -> x_3)[0],
					workspaceBytes, workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w3 matmul proj...\n");
		return -1;
	}


	printf("Submitting SwiGLU Activation...!\n");


	ret = submit_swiglu(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, ffn_dim, 
						activations -> x_1, activations -> x_3, activations -> x_temp_mlp);

	if (ret){
		fprintf(stderr, "Error: failed to submit swiglu activation...\n");
		return -1;
	}


	printf("Submitting FFN w2 matmul...!\n");

	ret = submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, fwd_dt, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q, 
					1.0, 1.0,
					transformer_block -> w_2, activations -> x_temp_mlp, activations -> x_o, activations -> x_layer_out,
					workspaceBytes, workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w2 matmul proj...\n");
		return -1;
	}


	return 0;

}


int submit_transformer_head(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                        void * X, Transformer_Head * transformer_head,
                        Transformer_Head_Activations * head_activations, 
						void * logits_out) {

    int ret;

    // Get dimensions from embedding config
    int vocab_size = (transformer_head -> embedding_config).vocab_size;
    int embedding_size = (transformer_head -> embedding_config).embedding_size;

    // RMS Normalization
    ret = submit_rms_norm(dataflow_handle, compute_stream_id,
                         transformer_head -> fwd_dt,
                         head_activations -> num_tokens,
                         embedding_size, transformer_head -> eps,  // Get eps from transformer_head
                         transformer_head -> w_head_norm,  // Changed from w_out_norm to w_head_norm
                         X,
                         head_activations -> x_temp,
                         head_activations -> head_norm_weighted_sums,  // Store weighted sums for backprop
                         head_activations -> head_norm_rms_vals);     // Store RMS values for backprop
    if (ret) {
        fprintf(stderr, "Error: Failed to submit RMS normalization in transformer head...\n");
        return ret;
    }

    // Output projection
    // Input x_temp is in row-major format
    // Weights w_out are in col-major format
    // Want output in row-major format: head_out[num_tokens, vocab_size] = x_temp[num_tokens, embedding_size] @ w_out[embedding_size, vocab_size]
    // Use transa=1 for row-major input, transb=0 for col-major weights
    ret = submit_matmul(dataflow_handle, compute_stream_id,
                       transformer_head -> fwd_dt,
                       transformer_head -> fwd_dt,
                       DATAFLOW_NONE,
                       transformer_head -> fwd_dt,
                       transformer_head -> compute_dt,
                       1, 0,  // transa=1 for row-major input, transb=0 for col-major weights
                       embedding_size,    // m = embedding_size (due to transa=1)
                       vocab_size,        // n = vocab_size
                       head_activations -> num_tokens,  // k = num_tokens (due to transa=1)
                       1.0, 0.0,
                       transformer_head -> w_head,       // Changed from w_out to w_head
                       head_activations -> x_temp,      // B[num_tokens, embedding_size] in row-major
                       NULL,
                       head_activations -> head_out,    // C[num_tokens, vocab_size] in row-major
                       0, NULL);  // No workspace needed
    if (ret) {
        fprintf(stderr, "Error: Failed to submit output projection in transformer head...\n");
        return ret;
    }

    // Apply softmax over vocabulary dimension
    // Each row (corresponding to a token) should sum to 1
    ret = submit_softmax(dataflow_handle, compute_stream_id,
                        transformer_head -> fwd_dt,
                        transformer_head -> bwd_dt,  // Use bwd_dt for backward pass
                        head_activations -> num_tokens,  // Number of rows to process
                        vocab_size,                      // Size of softmax dimension
                        head_activations -> head_out,   // Input logits
                        logits_out);                    // Output probabilities to provided buffer
    if (ret) {
        fprintf(stderr, "Error: Failed to submit softmax in transformer head...\n");
        return ret;
    }

    return 0;
}


// dX_out is upstream gradient
// it is in col-major orientation

// now to do dX = matmul(dX_out, W^T)
// we remember that W is being stored in col-major as well
// thus we can correctly get dX in col major as:
// we can do matmul(W, dx_Out^T)
// This is because we will get a row major output of 
// dX^T = dX in col major

// To see how we get col major of dX recall:

// thus if we pass A = W in col-major = W^T effectively in row major
// and B = dX_Out in col-major,
// If we pass in M = w_in, K = w_out, N = x_in, which yields
// a (w_in, x_in) matrix which we interpret as col major dX




int submit_transformer_block_bwd_x(Dataflow_Handle * dataflow_handle, int compute_stream_id, 
                                 void * dX_out, void * X, Transformer_Block * transformer_block, 
                                 Transformer_Block_Activations * activations, 
                                 Transformer_Block_Activations * grad_activations) {

	int ret;
	DataflowDatatype bwd_dt = (transformer_block -> config).block_dt;
	DataflowDatatype compute_dt = (transformer_block -> config).compute_dt;

	int num_seqs = (activations -> config).num_seqs;
	int total_q = (activations -> config).total_q;
	int total_k = (activations -> config).total_k;
	
	int model_dim = (transformer_block -> config).model_dim;
	int kv_dim = (transformer_block -> config).kv_dim;
	int ffn_dim = (transformer_block -> config).ffn_dim;
	
	uint64_t workspaceBytes = (activations -> config).workspaceBytes;
	void * workspace = (activations -> config).workspace;

	int to_transa = 0;
	int to_transb = 0;

	// 1. Backprop through FFN w2 matmul
	// Forward: [num_tokens, ffn_dim] @ [ffn_dim, model_dim] -> [num_tokens, model_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = ffn_dim
	// K = output cols of dX = model_dim
	// N = batch dim = num_tokens
	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q,  // M = ffn_dim, K = model_dim, N = num_tokens
					1.0, 0.0,
					transformer_block -> w_2[0], dX_out, NULL, grad_activations -> x_temp_mlp,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w2 backward matmul...\n");
		return -1;
	}

	// 2. Backprop through SwiGLU
	ret = submit_swiglu_bwd_x(dataflow_handle, compute_stream_id,
						bwd_dt, bwd_dt,
						total_q, ffn_dim,
						activations -> x_1[0], activations -> x_3[0],
						grad_activations -> x_temp_mlp,
						grad_activations -> x_1[0], grad_activations -> x_3[0]);
	if (ret) {
		fprintf(stderr, "Error: failed to submit swiglu backward...\n");
		return -1;
	}

	// 3. Backprop through w1 and w3 matmuls
	// Forward: [num_tokens, model_dim] @ [model_dim, ffn_dim] -> [num_tokens, ffn_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = model_dim
	// K = output cols of dX = ffn_dim
	// N = batch dim = num_tokens
	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, ffn_dim, total_q,  // M = model_dim, K = ffn_dim, N = num_tokens
					1.0, 0.0,
					transformer_block -> w_1[0], grad_activations -> x_1[0], NULL, grad_activations -> x_temp,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w1 backward matmul...\n");
		return -1;
	}

	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, ffn_dim, total_q,  // M = model_dim, K = ffn_dim, N = num_tokens
					1.0, 1.0,  // Add to previous gradient
					transformer_block -> w_3[0], grad_activations -> x_3[0], NULL, grad_activations -> x_temp,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w3 backward matmul...\n");
		return -1;
	}

	// 4. Backprop through FFN RMS Norm
	ret = submit_rms_norm_bwd_x(dataflow_handle, compute_stream_id,
							bwd_dt, bwd_dt,
							total_q, model_dim, (transformer_block -> config).eps,
							activations -> ffn_norm_weighted_sums,
							activations -> ffn_norm_rms_vals,
							transformer_block -> w_ffn_norm,
							activations -> x_o,  // Input to norm
							grad_activations -> x_temp,  // Upstream gradient
							dX_out);                    // Final output gradient
	if (ret) {
		fprintf(stderr, "Error: failed to submit ffn norm backward...\n");
		return -1;
	}

	// 5. Backprop through attention output projection
	// Forward: [num_tokens, model_dim] @ [model_dim, model_dim] -> [num_tokens, model_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = model_dim
	// K = output cols of dX = model_dim
	// N = batch dim = num_tokens
	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q,  // M = model_dim, K = model_dim, N = num_tokens
					1.0, 0.0,
					transformer_block -> w_o, grad_activations -> x_o, NULL, grad_activations -> x_temp,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention output backward matmul...\n");
		return -1;
	}


	// ensure workspace is zerod out beforehand....

	ret = (dataflow_handle -> set_mem)(dataflow_handle, compute_stream_id, workspace, 0, workspaceBytes);
	if (ret){
		fprintf(stderr, "Error: unable to set attention workspace mem to 0 before submitting...\n");
		return -1;
	}

	// 6. Backprop through attention
	ret = submit_attention_bwd(dataflow_handle, compute_stream_id,
							bwd_dt,
							num_seqs, total_q, total_k,
							(activations -> config).q_seq_offsets,
							(activations -> config).q_seq_lens,
							(activations -> config).max_seqlen_q,
							(activations -> config).k_seq_offsets,
							(activations -> config).k_seq_lens,
							(activations -> config).max_seqlen_k,
							(transformer_block -> config).num_q_heads,
							(transformer_block -> config).num_kv_heads,
							(transformer_block -> config).head_dim,
							activations -> x_q,        // Q input
							activations -> x_k_local,  // K input
							activations -> x_v_local,  // V input
							activations -> x_temp,     // Attention output
							activations -> softmax_lse,// Softmax scaling factors
							grad_activations -> x_temp,// Upstream gradient
							grad_activations -> x_q,   // dQ output
							grad_activations -> x_k_local, // dK output
							grad_activations -> x_v_local, // dV output
							workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention backward...\n");
		return -1;
	}

	// 7. Backprop through RoPE
	ret = submit_rope_bwd_x(dataflow_handle, compute_stream_id,
						bwd_dt,
						(uint64_t)total_q * (uint64_t)model_dim,
						model_dim,
						(transformer_block -> config).head_dim,
						(transformer_block -> config).num_kv_heads,
						(transformer_block -> config).theta,
						(activations -> config).seq_positions,
						grad_activations -> x_q,
						grad_activations -> x_k_local);
	if (ret) {
		fprintf(stderr, "Error: failed to submit rope backward...\n");
		return -1;
	}

	// 8. Backprop through Q, K, V projections
	// Q Forward: [num_tokens, model_dim] @ [model_dim, model_dim] -> [num_tokens, model_dim]
	// Backward: dX = dY @ W^T
	// For Q:
	// M = output rows of dX = model_dim
	// K = output cols of dX = model_dim
	// N = batch dim = num_tokens
	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q,  // M = model_dim, K = model_dim, N = num_tokens
					1.0, 0.0,
					transformer_block -> w_q, grad_activations -> x_q, NULL, grad_activations -> x_temp,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit Q projection backward matmul...\n");
		return -1;
	}

	// For K/V:
	// Forward: [num_tokens, model_dim] @ [model_dim, kv_dim] -> [num_tokens, kv_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = model_dim
	// K = output cols of dX = kv_dim
	// N = batch dim = num_tokens
	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, kv_dim, total_q,  // M = model_dim, K = kv_dim, N = num_tokens
					1.0, 1.0,  // Add to previous gradient
					transformer_block -> w_k, grad_activations -> x_k_local, NULL, grad_activations -> x_temp,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit K projection backward matmul...\n");
		return -1;
	}

	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, kv_dim, total_q,  // M = model_dim, K = kv_dim, N = num_tokens
					1.0, 1.0,  // Add to previous gradient
					transformer_block -> w_v, grad_activations -> x_v_local, NULL, grad_activations -> x_temp,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit V projection backward matmul...\n");
		return -1;
	}

	// 9. Finally backprop through attention RMS norm
	ret = submit_rms_norm_bwd_x(dataflow_handle, compute_stream_id,
							bwd_dt, bwd_dt,
							total_q, model_dim, (transformer_block -> config).eps,
							activations -> attn_norm_weighted_sums,
							activations -> attn_norm_rms_vals,
							transformer_block -> w_attn_norm,
							X,  // Input to norm
							grad_activations -> x_temp,  // Upstream gradient
							dX_out);                    // Final output gradient
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention norm backward...\n");
		return -1;
	}

	return 0;
}

int submit_transformer_block_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id, 
								 void * dX_out, Transformer_Block_Activations * activations, 
								 Transformer_Block_Activations * grad_activations,
								 Transformer_Block * grad_weights) {
	
	int ret;
	DataflowDatatype bwd_dt = (grad_weights -> config).block_dt;
	DataflowDatatype compute_dt = (grad_weights -> config).compute_dt;

	int num_seqs = (activations -> config).num_seqs;
	int total_q = (activations -> config).total_q;
	int total_k = (activations -> config).total_k;
	
	int model_dim = (grad_weights -> config).model_dim;
	int kv_dim = (grad_weights -> config).kv_dim;
	int ffn_dim = (grad_weights -> config).ffn_dim;
	
	uint64_t workspaceBytes = (activations -> config).workspaceBytes;
	void * workspace = (activations -> config).workspace;

	int to_transa = 1;
	int to_transb = 0;

	// 1. FFN w2 weight gradients
	// dW2^T = x_temp_mlp @ dX_out^T
	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, bwd_dt, bwd_dt,  // Changed DATAFLOW_NONE to bwd_dt since we're accumulating
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q,
					1.0, 1.0,  // Changed beta to 1.0 to accumulate gradients
					dX_out, activations -> x_temp_mlp, grad_weights -> w_2, grad_weights -> w_2,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w2 weight gradient computation...\n");
		return -1;
	}

	// 2. FFN w1 and w3 weight gradients after SwiGLU
	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, bwd_dt, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q,
					1.0, 1.0,  // Accumulate gradients
					grad_activations -> x_1[0], activations -> x_temp, grad_weights -> w_1, grad_weights -> w_1,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w1 weight gradient computation...\n");
		return -1;
	}

	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, bwd_dt, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q,
					1.0, 1.0,  // Accumulate gradients
					grad_activations -> x_3[0], activations -> x_temp, grad_weights -> w_3, grad_weights -> w_3,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w3 weight gradient computation...\n");
		return -1;
	}

	// 3. FFN RMS Norm weight gradients
	// Note: rms_norm_bwd_w already accumulates gradients internally
	ret = submit_rms_norm_bwd_w(dataflow_handle, compute_stream_id,
							bwd_dt, bwd_dt,
							total_q, model_dim, (grad_weights -> config).eps,
							activations -> ffn_norm_rms_vals,
							activations -> x_o,
							grad_activations -> x_temp,
							grad_weights -> w_ffn_norm);
	if (ret) {
		fprintf(stderr, "Error: failed to submit ffn norm weight gradient computation...\n");
		return -1;
	}

	// 4. Attention output projection weight gradients
	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, bwd_dt, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q,
					1.0, 1.0,  // Accumulate gradients
					grad_activations -> x_o, activations -> x_temp, grad_weights -> w_o, grad_weights -> w_o,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention output weight gradient computation...\n");
		return -1;
	}

	// 5. Q, K, V projection weight gradients
	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, bwd_dt, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q,
					1.0, 1.0,  // Accumulate gradients
					grad_activations -> x_q, activations -> x_temp, grad_weights -> w_q, grad_weights -> w_q,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit Q projection weight gradient computation...\n");
		return -1;
	}

	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, bwd_dt, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					kv_dim, model_dim, total_q,
					1.0, 1.0,  // Accumulate gradients
					grad_activations -> x_k_local, activations -> x_temp, grad_weights -> w_k, grad_weights -> w_k,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit K projection weight gradient computation...\n");
		return -1;
	}

	ret = submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, bwd_dt, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					kv_dim, model_dim, total_q,
					1.0, 1.0,  // Accumulate gradients
					grad_activations -> x_v_local, activations -> x_temp, grad_weights -> w_v, grad_weights -> w_v,
					workspaceBytes, workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit V projection weight gradient computation...\n");
		return -1;
	}

	// 6. Attention RMS Norm weight gradients
	// Note: rms_norm_bwd_w already accumulates gradients internally
	ret = submit_rms_norm_bwd_w(dataflow_handle, compute_stream_id,
							bwd_dt, bwd_dt,
							total_q, model_dim, (grad_weights -> config).eps,
							activations -> attn_norm_rms_vals,
							activations -> x_temp,
							grad_activations -> x_temp,
							grad_weights -> w_attn_norm);
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention norm weight gradient computation...\n");
		return -1;
	}

	return 0;
}


int submit_transformer_head_bwd_x(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                        void * logits_out, uint32_t * labels, Transformer_Head * transformer_head,
                        Transformer_Head_Activations * head_activations,
                        Transformer_Head_Activations * grad_head_activations,
                        void * dX_out) {

    int ret;

    // Get dimensions from embedding config
    int vocab_size = (transformer_head -> embedding_config).vocab_size;
    int embedding_size = (transformer_head -> embedding_config).embedding_size;

    // First compute cross entropy loss gradients
    // This will subtract 1.0 from the correct class logits in-place
    ret = submit_cross_entropy_loss(dataflow_handle, compute_stream_id,
                                  transformer_head -> bwd_dt,
                                  head_activations -> num_tokens,  // Number of rows (tokens)
                                  vocab_size,                      // Number of columns (vocab size)
                                  logits_out,                      // Predicted logits
                                  labels);                         // Ground truth labels
    if (ret) {
        fprintf(stderr, "Error: Failed to submit cross entropy loss in transformer head backward...\n");
        return ret;
    }

    // Now backpropagate through the output projection
    // Input logits_out is in row-major format after cross entropy
    // Weights w_head are in col-major format
    // Want output in row-major format: dx_temp = dlogits @ w_head^T
    // For backward pass with transa=0, transb=0:
    // M = fwd_K = embedding_size
    // K = fwd_N = vocab_size
    // N = fwd_M = num_tokens
    ret = submit_matmul(dataflow_handle, compute_stream_id,
                       transformer_head -> bwd_dt,
                       transformer_head -> bwd_dt,
                       DATAFLOW_NONE,
                       transformer_head -> bwd_dt,
                       transformer_head -> compute_dt,
                       0, 0,  // transa=0 for row-major dlogits, transb=0 for col-major weights
                       embedding_size,                  // m = embedding_size (fwd_K)
                       head_activations -> num_tokens,  // n = num_tokens (fwd_M)
                       vocab_size,                      // k = vocab_size (fwd_N)
                       1.0, 0.0,
                       logits_out,                     // dlogits[num_tokens, vocab_size] in row-major
                       transformer_head -> w_head,      // w_head[embedding_size, vocab_size] in col-major
                       NULL,
                       grad_head_activations -> x_temp, // dx_temp[num_tokens, embedding_size] in row-major
                       0, NULL);  // No workspace needed
    if (ret) {
        fprintf(stderr, "Error: Failed to submit output projection backward in transformer head...\n");
        return ret;
    }

    // Finally backpropagate through RMS normalization
    ret = submit_rms_norm_bwd_x(dataflow_handle, compute_stream_id,
                               transformer_head -> bwd_dt,
                               transformer_head -> bwd_dt,
                               head_activations -> num_tokens,
                               embedding_size,
                               transformer_head -> eps,
                               head_activations -> head_norm_weighted_sums,
                               head_activations -> head_norm_rms_vals,
                               transformer_head -> w_head_norm,
                               head_activations -> head_out,         // Original input
                               grad_head_activations -> x_temp,      // Upstream gradient
                               dX_out);                             // Final output gradient
    if (ret) {
        fprintf(stderr, "Error: Failed to submit RMS norm backward in transformer head...\n");
        return ret;
    }

    return 0;
}

int submit_transformer_head_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                         Transformer_Head_Activations * head_activations,
                         Transformer_Head_Activations * grad_head_activations,
                         Transformer_Head * grad_transformer_head) {

    int ret;

    // Get dimensions from embedding config
    int vocab_size = (grad_transformer_head -> embedding_config).vocab_size;
    int embedding_size = (grad_transformer_head -> embedding_config).embedding_size;

    // Get workspace from head_activations
    uint64_t workspaceBytes = grad_head_activations -> workspaceBytes;
    void * workspace = grad_head_activations -> workspace;

    // 1. RMS Norm weight gradients
    // Note: rms_norm_bwd_w already accumulates gradients internally
    ret = submit_rms_norm_bwd_w(dataflow_handle, compute_stream_id,
                               grad_transformer_head -> fwd_dt,
                               grad_transformer_head -> bwd_dt,
                               head_activations -> num_tokens,
                               embedding_size,
                               grad_transformer_head -> eps,
                               head_activations -> head_norm_rms_vals,  // RMS values from forward pass
                               head_activations -> head_out,            // Original input
                               grad_head_activations -> x_temp,         // Upstream gradient
                               grad_transformer_head -> w_head_norm);   // Output gradient
    if (ret) {
        fprintf(stderr, "Error: Failed to submit RMS norm weight gradient computation in transformer head...\n");
        return ret;
    }

    // 2. Output projection weight gradients
    // Forward: [num_tokens, embedding_size] @ [embedding_size, vocab_size] -> [num_tokens, vocab_size]
    // Backward for weights: dW = X^T @ dY
    // For matmul with transa=1, transb=0:
    // M = embedding_size (rows of dW)
    // K = num_tokens (reduction dim)
    // N = vocab_size (cols of dW)
    ret = submit_matmul(dataflow_handle, compute_stream_id,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> compute_dt,
                       1, 0,  // transa=1 for X^T, transb=0 for dY
                       embedding_size, vocab_size, head_activations -> num_tokens,  // M, K, N
                       1.0, 1.0,  // Accumulate gradients
                       head_activations -> x_temp,           // Input activations [num_tokens, embedding_size]
                       grad_head_activations -> head_out,    // Gradient of output [num_tokens, vocab_size]
                       grad_transformer_head -> w_head,      // Previous gradient
                       grad_transformer_head -> w_head,      // Output gradient
                       workspaceBytes, workspace);          // Use workspace from grad_head_activations
    if (ret) {
        fprintf(stderr, "Error: Failed to submit output projection weight gradient computation in transformer head...\n");
        return ret;
    }

    return 0;
}