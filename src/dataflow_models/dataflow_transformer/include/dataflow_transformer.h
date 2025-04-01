#ifndef DATAFLOW_TRANSFORMER_H
#define DATAFLOW_TRANSFORMER_H

#include "dataflow_handle.h"
#include "dataflow_models.h"
#include "dataflow_ops.h"


typedef struct transformer_block_weight_offsets {
	uint64_t w_attn_norm;
	uint64_t w_q;
	uint64_t w_k;
	uint64_t w_v;
	uint64_t w_o;
	uint64_t w_ffn_norm;
	uint64_t w_router;
	// arrays of length num_local experts if DATAFLOW_MOE_MLP
	// otherwise arrays of length 1
	uint64_t * w_1;
	uint64_t * w_2;
	uint64_t * w_3;
} Transformer_Block_Weight_Offsets;

typedef struct transformer_block_config {
	DataflowDatatype block_dt;
	// for Matmul accumulation
	// if on Geforce using FP16 gives twice as much perf.
	DataflowDatatype compute_dt;
	DataflowNormalizationType normalization_type;
	DataflowAttentionType attention_type;
	DataflowMLPType mlp_type;
	DataflowActivationType activation_type;

	float eps;
	int theta;

	int num_q_heads;
	int num_kv_heads;
	int head_dim;

	// We set these
	// model_dim = num_q_heads * head_dim
	// kv_dim = num_kv_heads * head_dim
	int model_dim;
	int kv_dim;

	// if using mlp_type = DATAFLOW_MOE_MLP
	// this ffn_dim is the dimension of each expert
	int ffn_dim;

	// for models with mlp_type = DATAFLOW_MOE_MLP
	MoE_Config moe_config;
		
	// each unqiue weight pointer
	// must be a multiple of this value
	// (e.g. 256 to use tensor cores, or maybe 512 to load direct from SSD, or maybe 4k to start on unique sys ram pages, etc...)
	int pointer_alignment;

	Transformer_Block_Weight_Offsets weight_offsets;
	// the total amount of memory required
	// to hold all weights
	uint64_t block_raw_size;
	
	// the total amount of memory
	uint64_t block_aligned_size;
	
} Transformer_Block_Config;

typedef struct transformer_block {
	Transformer_Block_Config config;
	void * buffer;
	// offsets into buffer, based upon
	// config.weight_offsets
	void * w_attn_norm;
	void * w_q;
	void * w_k;
	void * w_v;
	void * w_o;
	void * w_ffn_norm;
	// if non-moe this is null
	void * w_router;
	// each of these 
	// are arrays of length = MAX(config.num_local_experts, 1)
	// (if non-moe, just use index 0)

	// if DATAFLOW_VANILLA_MLP, then w_3 should all be null
	void ** w_1;
	void ** w_2;
	void ** w_3;
} Transformer_Block;


typedef struct transformer_block_activations_config{
	int num_seqs;

	// will be sum of q_seq_lens
	int total_q;
	// will be sum of k_seq_lens
	int total_k;

	// Device Metadata Buffers

	// for use in both attention and matmuls
	void * workspace;
	uint64_t workspaceBytes;
	
	// for rope
	// of size total_q
	int * seq_positions;

	// for attn

	// of size num_seqs + 1
	// where index i represents starting
	// token offset of seq i. The value at q_seq_offsets[num_seqs] 
	// shoudl be q_seq_offsets[num_seqs - 1] + q_seq_lens[num_seqs - 1]
	int * q_seq_offsets;
	// of size num_seqs
	// q_seq_lens[i] represents total new queries to process for seq i,
	// starting at the corresponding offset, consecutively
	int * q_seq_lens;
	// largest value from q_seq_lens
	int max_seqlen_q;

	// of size num_seqs + 1; similar to q_seq_offsets, but now represents for kv cache
	// starting offsets. We can pass x_k, and x_v of different number of total
	// tokens (usually >= total_q) so during forwards pass the new queries
	// can utilize prior cached keys and values

	// during backwards pass if we start processing chunks from the end of the 
	// seq to the beginning, then we can also use this in order to  
	int * k_seq_offsets;
	// of size num_seqs; similar to q_seq_lens
	// Contains the number of keys we want to use for each sequence,
	// starting at the offset, consecutively
	int * k_seq_lens;
	// largest value from k_seq_lens
	int max_seqlen_k;

	// Can also add in metadata regarding MoE if needed...
} Transformer_Block_Activations_Config;

typedef struct transformer_block_activations {
	Transformer_Block * block;
	Transformer_Block_Activations_Config config;
	void * buffer;


	// used during backprop
	void * attn_norm_weighted_sums;
	void * attn_norm_rms_vals;
	void * x_q;
	// These are the outputs of passing
	// normalized input through K and V weight
	// matrices
	void * x_k_local;
	void * x_v_local;

	// softmax_lse
	void * softmax_lse;
	void * x_o;
	// used during backprop
	void * ffn_norm_weighted_sums;
	void * ffn_norm_rms_vals;
	void ** x_1;
	void ** x_2;
	void ** x_3;
	void * x_layer_out;

	// used as temporary buffer during
	// norm outputs and attention output
	void * x_temp;

	// used as temporary output buffer during
	// MLP

	// needs to be total_q * ffn_dim
	void * x_temp_mlp;

	// Can use copy_to_seq_cache to move x_k_local (post rope)
	// and x_v_local to proper locations within these matrices
	// if there are multiple seqs and also prior caching involved...

	// These are the matrices passed to attention
	// mechanism and may contain other state
	// (i.e. prior computed cached keys/values during fwd,
	// 		or accumulated gradients during backprop)
	void * x_k;
	void * x_v;
} Transformer_Block_Activations;


Transformer_Block * init_transformer_block(DataflowDatatype block_dt, DataflowDatatype compute_dt,
						   DataflowNormalizationType normalization_type, 
						   DataflowAttentionType attention_type,
						   DataflowMLPType mlp_type,
						   DataflowActivationType activation_type,
						   float eps, int theta,
						   int num_q_heads, int num_kv_heads, int head_dim,
						   int ffn_dim,
						   MoE_Config * moe_config,
						   int pointer_alignment);

// first init must be called
uint64_t get_transformer_block_raw_size(Transformer_Block * transformer_block);
uint64_t get_transformer_block_aligned_size(Transformer_Block * transformer_block);

// now pass in a buffer of size >= size specified above
// and the pointers will be properly assigned (ensuring alignment)
int bind_transformer_block(void * buffer, Transformer_Block * transformer_block);


// the file consists of all weights 
//int load_transformer_block(char * filename, Transformer_Block * transformer_block);


// Need to set Seq Batch metadata...!
//int bind_transformer_block_activations(void * buffer, Seq_Batch * seq_batch, Transformer_Block * block, Transformer_Block_Activations * activation_buffer);


int submit_transformer_block(Dataflow_Handle * dataflow_handle, int compute_stream_id, void * X, Transformer_Block * transformer_block, Transformer_Block_Activations * activations);

int submit_transformer_block_bwd_x(Dataflow_Handle * dataflow_handle, int compute_stream_id, void * dX, Transformer_Block * transformer_block, Transformer_Block_Activations * activations, Transformer_Block_Activations * grad_activations);

int submit_transformer_block_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id, Transformer_Block * transformer_block, Transformer_Block_Activations * grad_activations, Transformer_Block * grad_weights);


#endif