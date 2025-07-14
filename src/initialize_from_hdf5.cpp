#include "Rcpp.h"
#include "Rtatami.h"
#include "tatami_hdf5/tatami_hdf5.hpp"

#include <string>

//[[Rcpp::export(rng=false)]]
SEXP initialize_from_hdf5_sparse(std::string file, std::string name, int nrow, int ncol, bool csr, int cache_size) {
    tatami_hdf5::CompressedSparseMatrixOptions opt;
    opt.maximum_cache_size = cache_size;
    auto output = Rtatami::new_BoundNumericMatrix();
    output->ptr.reset(new tatami_hdf5::CompressedSparseMatrix<double, int>(nrow, ncol, std::move(file), name + "/data", name + "/indices", name + "/indptr", csr, opt));
    return output; 
}

//[[Rcpp::export(rng=false)]]
SEXP initialize_from_hdf5_dense(std::string file, std::string name, bool transpose, int cache_size) {
    tatami_hdf5::DenseMatrixOptions opt;
    opt.maximum_cache_size = cache_size;
    auto output = Rtatami::new_BoundNumericMatrix();
    output->ptr.reset(new tatami_hdf5::DenseMatrix<double, int>(std::move(file), std::move(name), transpose, opt));
    return output; 
}

