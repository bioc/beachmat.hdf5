#include "Rcpp.h"
#include "Rtatami.h"
#include "tatami_hdf5/tatami_hdf5.hpp"
#include "sanisizer/sanisizer.hpp"

#include <string>
#include <cstdint>
#include <limits>
#include <vector>

template<typename Tx, typename Ti>
SEXP load_into_memory_sparse_base(const std::string& file, const std::string& name, int nrow, int ncol, bool csr) {
    auto output = Rtatami::new_BoundNumericMatrix();
    output->ptr = tatami_hdf5::load_compressed_sparse_matrix<double, int, std::vector<Tx>, std::vector<Ti> >(
        nrow, 
        ncol, 
        file, 
        name + "/data", 
        name + "/indices", 
        name + "/indptr",
        csr
    );
    return output;
}

template<typename Tx>
SEXP load_into_memory_sparse_i_max(const std::string& file, const std::string& name, int nrow, int ncol, bool csr) {
    if (sanisizer::is_less_than_or_equal(csr ? ncol : nrow, std::numeric_limits<std::uint16_t>::max())) {
        return load_into_memory_sparse_base<Tx, std::uint16_t>(file, name, nrow, ncol, csr);
    } else {
        return load_into_memory_sparse_base<Tx, int>(file, name, nrow, ncol, csr);
    }
}

// TODO: migrate internal type choices to tatami_hdf5.
std::pair<bool, bool> check_type(const std::string& file, const std::string& name) {
    H5::H5File handle(file, H5F_ACC_RDONLY);
    auto xhandle = handle.openDataSet(name);
    auto xtype = xhandle.getDataType();

    bool is_ushort = false;
    bool is_float = (xtype.getClass() == H5T_FLOAT);
    if (!is_float) {
        H5::IntType xitype(xhandle);
        is_ushort = (xitype.getSize() <= 2 && xitype.getSign() == H5T_SGN_NONE);
    }

    return std::make_pair(is_float, is_ushort);
}

//[[Rcpp::export(rng=false)]]
SEXP load_into_memory_sparse(std::string file, std::string name, int nrow, int ncol, bool csr, bool forced_int) {
    auto inspected = check_type(file, name + "/data");
    auto is_float = inspected.first;
    auto is_ushort = inspected.second;

    if (is_float && !forced_int) {
        return load_into_memory_sparse_i_max<double>(file, name, nrow, ncol, csr);
    } else if (is_ushort) {
        return load_into_memory_sparse_i_max<std::uint16_t>(file, name, nrow, ncol, csr);
    } else {
        return load_into_memory_sparse_i_max<int>(file, name, nrow, ncol, csr);
    }
}

//[[Rcpp::export(rng=false)]]
SEXP load_into_memory_dense(std::string file, std::string name, bool forced_int, bool transpose) {
    auto inspected = check_type(file, name);
    auto is_float = inspected.first;
    auto is_ushort = inspected.second;

    auto output = Rtatami::new_BoundNumericMatrix();
    if (is_float && !forced_int) {
        output->ptr = tatami_hdf5::load_dense_matrix<double, int, std::vector<double> >(file, name, transpose);
    } else if (is_ushort) {
        output->ptr = tatami_hdf5::load_dense_matrix<double, int, std::vector<std::uint16_t> >(file, name, transpose);
    } else {
        output->ptr = tatami_hdf5::load_dense_matrix<double, int, std::vector<int> >(file, name, transpose);
    }
    return output;
}

template<typename Tx>
SEXP load_into_memory_dense_to_sparse_base(const std::string& file, const std::string& name, bool transpose, int cache_size, bool byrow) {
    tatami_hdf5::DenseMatrixOptions opt;
    opt.maximum_cache_size = cache_size;
    tatami_hdf5::DenseMatrix<double, int> mat(file, name, transpose, opt);

    auto output = Rtatami::new_BoundNumericMatrix();
    if (sanisizer::is_less_than_or_equal(byrow ? mat.ncol() : mat.nrow(), std::numeric_limits<std::uint16_t>::max())) {
        output->ptr = tatami::convert_to_compressed_sparse<double, int, Tx, std::uint16_t>(&mat, byrow);
    } else {
        output->ptr = tatami::convert_to_compressed_sparse<double, int, Tx, int>(&mat, byrow);
    }
    return output;
}

//[[Rcpp::export(rng=false)]]
SEXP load_into_memory_dense_as_sparse(std::string file, std::string name, bool forced_int, bool transpose, int cache_size, bool byrow) {
    auto inspected = check_type(file, name);
    auto is_float = inspected.first;
    auto is_ushort = inspected.second;

    if (is_float && !forced_int) {
        return load_into_memory_dense_to_sparse_base<double>(file, name, transpose, cache_size, byrow);
    } else if (is_ushort) {
        return load_into_memory_dense_to_sparse_base<std::uint16_t>(file, name, transpose, cache_size, byrow);
    } else {
        return load_into_memory_dense_to_sparse_base<int>(file, name, transpose, cache_size, byrow);
    }
}
