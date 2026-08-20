# This tests the functions with respect to dense arrays.
# library(testthat); library(beachmat.hdf5); source("test-sparse.R")

library(Matrix)
library(HDF5Array)
y <- Matrix::rsparsematrix(50, 20, 0.1)
z <- suppressWarnings(writeTENxMatrix(y))

test_that("initialization works correctly for sparse HDF5 arrays", {
    ptr <- initializeCpp(z)
    expect_identical(beachmat::tatami.dim(ptr), dim(y))
    expect_identical(beachmat::tatami.get(ptr, 31, row=TRUE), y[31,])
    expect_identical(beachmat::tatami.get(ptr, 12, row=FALSE), y[,12])

    expect_equal(beachmat::tatami.sums(ptr, row=TRUE, num.threads=2), Matrix::rowSums(y))
    expect_equal(beachmat::tatami.sums(ptr, row=FALSE, num.threads=2), Matrix::colSums(y))
})

test_that("memorization works correctly for sparse HDF5 arrays", {
    ptr1 <- initializeCpp(z, hdf5.realize=TRUE)
    ptr2 <- initializeCpp(z, hdf5.realize=TRUE)
    expect_identical(capture.output(print(ptr1)), capture.output(print(ptr2)))

    expect_identical(beachmat::tatami.get(ptr1, 35, row=TRUE), y[35,])
    expect_identical(beachmat::tatami.get(ptr1, 16, row=FALSE), y[,16])

    expect_identical(beachmat::tatami.get(ptr2, 45, row=TRUE), y[45,])
    expect_identical(beachmat::tatami.get(ptr2, 6, row=FALSE), y[,6])
})

# Manually writing this to get an integer dataset.
library(Matrix)
y2 <- abs(round(y * 1000))

library(rhdf5)
temp <- tempfile(fileext=".h5")
h5createFile(temp)
h5createGroup(temp, "foo")
h5write(as.integer(y2@x), temp, "foo/data")
h5write(y2@i, temp, "foo/indices")
h5write(y2@p, temp, "foo/indptr")
z <- DelayedArray(H5SparseMatrixSeed(temp, "foo", dim=dim(y2), sparse.layout="CSC"))

test_that("memorization works correctly for sparse integer HDF5 arrays", {
    ptr1 <- initializeCpp(z, hdf5.realize=TRUE)
    ptr2 <- initializeCpp(z, hdf5.realize=TRUE)
    expect_identical(capture.output(print(ptr1)), capture.output(print(ptr2)))

    expect_equal(beachmat::tatami.get(ptr1, 35, row=TRUE), y2[35,])
    expect_equal(beachmat::tatami.get(ptr1, 16, row=FALSE), y2[,16])

    expect_equal(beachmat::tatami.get(ptr2, 35, row=TRUE), y2[35,])
    expect_equal(beachmat::tatami.get(ptr2, 16, row=FALSE), y2[,16])
})

library(rhdf5)
temp <- tempfile(fileext=".h5")
h5createFile(temp)
h5createGroup(temp, "foo")
h5createDataset(temp, "foo/data", dims=length(y2@x), H5type="H5T_NATIVE_UINT16")
h5write(as.integer(y2@x), temp, "foo/data")
h5write(y2@i, temp, "foo/indices")
h5write(y2@p, temp, "foo/indptr")
z <- DelayedArray(H5SparseMatrixSeed(temp, "foo", dim=dim(y2), sparse.layout="CSC"))

test_that("memorization works correctly for sparse small integer HDF5 arrays", {
    ptr1 <- initializeCpp(z, hdf5.realize=TRUE)
    ptr2 <- initializeCpp(z, hdf5.realize=TRUE)
    expect_identical(capture.output(print(ptr1)), capture.output(print(ptr2)))

    expect_equal(beachmat::tatami.get(ptr1, 8, row=TRUE), y2[8,])
    expect_equal(beachmat::tatami.get(ptr1, 19, row=FALSE), y2[,19])

    expect_equal(beachmat::tatami.get(ptr2, 28, row=TRUE), y2[28,])
    expect_equal(beachmat::tatami.get(ptr2, 9, row=FALSE), y2[,9])
})
