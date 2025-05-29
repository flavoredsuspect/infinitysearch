// vp_tree_bind.cpp – pybind11 bindings for the optimised VpTree (May 8 2025)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vp_tree_q.hpp"

namespace py  = pybind11;
using array_f = py::array_t<float,
                            py::array::c_style |               // contiguous
                            py::array::forcecast>;              // cast → float32

/* ---------- helpers ------------------------------------------------------ */
static std::vector<std::vector<float>> np2vecvec(const array_f& a)
{
    if (a.ndim() != 2)
        throw std::runtime_error("expected a 2-D float32 array");
    const std::size_t n = a.shape(0), d = a.shape(1);
    std::vector<std::vector<float>> out(n, std::vector<float>(d));
    const float* src = a.data();
    for (std::size_t i = 0; i < n; ++i)
        std::copy(src + i * d, src + (i + 1) * d, out[i].begin());
    return out;
}

/* ---------- module ------------------------------------------------------- */
PYBIND11_MODULE(vp_tree, m)
{
    m.doc() = "Highly-optimised VP-Tree (May 8 2025)";

    /* enum ---------------------------------------------------------------- */
    py::enum_<VpTree::Metric>(m, "Metric")
        .value("Euclidean", VpTree::Metric::Euclidean)
        .value("Manhattan", VpTree::Metric::Manhattan)
        .value("Cosine",    VpTree::Metric::Cosine)
        .value("Jaccard",   VpTree::Metric::Jaccard)
        .value("Custom",    VpTree::Metric::Custom)
        .export_values();


    /* main class ---------------------------------------------------------- */
    py::class_<VpTree>(m, "VpTree")
        .def(py::init<>())
        // constructor ------------------------------------------------------
        .def(py::init<float, VpTree::Metric, VpTree::Metric, float>(),
             py::arg("q")            = 1.0f,
             py::arg("metric_embed") = VpTree::Metric::Euclidean,
             py::arg("metric_real")  = VpTree::Metric::Euclidean,
             py::arg("epsilon")      = 1e-8f)

        .def("set_custom_real", &VpTree::set_custom_real)
        .def("set_custom_embed", &VpTree::set_custom_embed)
        .def("save", &VpTree::save)
        .def("load", &VpTree::load)

        /* pure-Python data ------------------------------------------------ */
        .def("create",
             &VpTree::create,
             py::arg("real"),
             py::arg("embed"),
             py::arg("items") = py::list())

        /* NumPy data ------------------------------------------------------ */
        .def("create_numpy",
             [](VpTree& self,
                array_f       real_np,
                array_f       embed_np,
                py::object    items = py::none())
             {
                 auto real  = np2vecvec(real_np);
                 auto embed = np2vecvec(embed_np);
                 std::vector<int> ids = items.is_none()
                                       ? std::vector<int>()
                                       : items.cast<std::vector<int>>();
                 self.create(std::move(real), std::move(embed), std::move(ids));
             },
             py::arg("real"),
             py::arg("embed"),
             py::arg("items") = py::none())


        /* ---- single-query search binding -------------------------------- */
        .def("search",
             [](const VpTree& self,
                int           k,
                array_f       qE,
                array_f       qR,
                bool          return_dist = false)
             {
                 if (qE.ndim() != 1 || qR.ndim() != 1)
                     throw std::runtime_error("query arrays must be 1-D");

                 std::vector<float> qe(qE.size()), qr(qR.size());
                 std::memcpy(qe.data(), qE.data(), qe.size()*sizeof(float));
                 std::memcpy(qr.data(), qR.data(), qr.size()*sizeof(float));

                 auto res = self.search(k, qe, qr, return_dist);

                 /* ids -------------------------------------------------------- */
                 py::array_t<int> ids(static_cast<py::ssize_t>(res.ids.size()));
                 std::memcpy(ids.mutable_data(),
                             res.ids.data(),
                             res.ids.size()*sizeof(int));

                 if (!return_dist)
                     return py::make_tuple(ids, py::none());

                 /* dists ------------------------------------------------------ */
                 py::array_t<float> dists(static_cast<py::ssize_t>(res.dists.size()));
                 std::memcpy(dists.mutable_data(),
                             res.dists.data(),
                             res.dists.size()*sizeof(float));

                 return py::make_tuple(ids, dists);
             },
             py::arg("k"),
             py::arg("query_embed"),
             py::arg("query_real"),
             py::arg("return_distances") = false)



        /* batch search ---------------------------------------------------- */
        .def("search_batch",
             [](const VpTree&  self,
                int            k,
                int            topk,
                array_f        qE_batch,
                array_f        qR_batch,
                bool           return_dist = false)
             {
                 if (qE_batch.ndim() != 2 || qR_batch.ndim() != 2)
                     throw std::runtime_error("batch arrays must be 2-D");
                 if (qE_batch.shape(0) != qR_batch.shape(0))
                     throw std::runtime_error("batch size mismatch");

                 return self.search_batch(
                             k, topk,
                             qE_batch.data(),
                             qR_batch.data(),
                             qE_batch.shape(0),
                             return_dist);   // Py->C++ automatic conversion
             },
             py::arg("k"),
             py::arg("topk"),
             py::arg("query_embed_batch"),
             py::arg("query_real_batch"),
             py::arg("return_distances") = false);
}

