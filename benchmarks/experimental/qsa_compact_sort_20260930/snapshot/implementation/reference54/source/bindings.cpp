// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

at::Tensor flash_attention_grouped_sparse_page4_plan(
    const at::Tensor& logical_indices, const at::Tensor& block_table,
    const at::Tensor& token_to_req, const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths, at::Tensor& output_blocks,
    at::Tensor& output_masks, at::Tensor& output_seq_lens, int page_size,
    int physical_page_stride, int num_cache_blocks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("plan_fwd", &flash_attention_grouped_sparse_page4_plan,
             "Plan grouped sparse page4 metadata (SM70)",
             pybind11::arg("logical_indices"),
             pybind11::arg("block_table"), pybind11::arg("token_to_req"),
             pybind11::arg("query_positions"),
             pybind11::arg("sequence_lengths"),
             pybind11::arg("output_blocks"), pybind11::arg("output_masks"),
             pybind11::arg("output_seq_lens"), pybind11::arg("page_size"),
             pybind11::arg("physical_page_stride"),
             pybind11::arg("num_cache_blocks"));
}
