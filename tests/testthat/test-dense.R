# This tests the functions with respect to dense arrays.
# library(testthat); library(beachmat.hdf5); source("test-dense.R")

library(HDF5Array)
y <- matrix(runif(1000), ncol=20, nrow=50)
z <- as(y, "HDF5Array")

test_that("initialization works correctly for dense HDF5 arrays", {
    ptr <- initializeCpp(z)
    expect_identical(beachmat::tatami.dim(ptr), dim(y))
    expect_identical(beachmat::tatami.get(ptr, 1, row=TRUE), y[1,])
    expect_identical(beachmat::tatami.get(ptr, 2, row=FALSE), y[,2])

    expect_equal(beachmat:::tatami.sums(ptr, num.threads=2, row=TRUE), rowSums(y))
    expect_equal(beachmat:::tatami.sums(ptr, num.threads=2, row=FALSE), colSums(y))
})

test_that("memorization works correctly for dense HDF5 arrays", {
    ptr1 <- initializeCpp(z, hdf5.realize=TRUE)
    ptr2 <- initializeCpp(z, hdf5.realize=TRUE)
    expect_identical(capture.output(print(ptr1)), capture.output(print(ptr2)))

    expect_identical(beachmat::tatami.get(ptr1, 5, row=TRUE), y[5,])
    expect_identical(beachmat::tatami.get(ptr1, 6, row=FALSE), y[,6])

    expect_identical(beachmat::tatami.get(ptr2, 5, row=TRUE), y[5,])
    expect_identical(beachmat::tatami.get(ptr2, 6, row=FALSE), y[,6])
})

library(Matrix)
y <- as.matrix(Matrix::rsparsematrix(50, 20, 0.1))
z <- as(y, "HDF5Array")

test_that("memorization works correctly for dense-as-sparse HDF5 arrays", {
    ptr1 <- initializeCpp(z, hdf5.realize=TRUE)
    ptr2 <- initializeCpp(z, hdf5.realize=TRUE)
    expect_identical(capture.output(print(ptr1)), capture.output(print(ptr2)))

    expect_identical(beachmat::tatami.get(ptr1, 45, row=TRUE), y[45,])
    expect_identical(beachmat::tatami.get(ptr1, 16, row=FALSE), y[,16])

    expect_identical(beachmat::tatami.get(ptr2, 45, row=TRUE), y[45,])
    expect_identical(beachmat::tatami.get(ptr2, 16, row=FALSE), y[,16])
})

library(Matrix)
y <- matrix(sample(1000), 40L, 25L)
z <- as(y, "HDF5Array")

test_that("memorization works correctly for integer HDF5 arrays", {
    ptr1 <- initializeCpp(z, hdf5.realize=TRUE)
    ptr2 <- initializeCpp(z, hdf5.realize=TRUE)
    expect_identical(capture.output(print(ptr1)), capture.output(print(ptr2)))

    expect_equal(beachmat::tatami.get(ptr1, 35, row=TRUE), y[35,])
    expect_equal(beachmat::tatami.get(ptr1, 16, row=FALSE), y[,16])

    expect_equal(beachmat::tatami.get(ptr2, 35, row=TRUE), y[35,])
    expect_equal(beachmat::tatami.get(ptr2, 16, row=FALSE), y[,16])
})

library(Matrix)
z <- writeHDF5Array(y, H5type="H5T_NATIVE_UINT16")

test_that("memorization works correctly for small integer HDF5 arrays", {
    ptr1 <- initializeCpp(z, hdf5.realize=TRUE)
    ptr2 <- initializeCpp(z, hdf5.realize=TRUE)
    expect_identical(capture.output(print(ptr1)), capture.output(print(ptr2)))

    expect_equal(beachmat::tatami.get(ptr1, 8, row=TRUE), y[8,])
    expect_equal(beachmat::tatami.get(ptr1, 19, row=FALSE), y[,19])

    expect_equal(beachmat::tatami.get(ptr2, 8, row=TRUE), y[8,])
    expect_equal(beachmat::tatami.get(ptr2, 19, row=FALSE), y[,19])
})
