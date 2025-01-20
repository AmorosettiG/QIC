! Assignment 2, Exercise 3 (file 2)
! Gabriel Amorosetti

program main

    use mod_matrix_c8
    implicit none

    type(complex8_matrix) :: A, C ! Defining the complex matrices A and its adjoint C
    complex*8 :: x                ! Trace of the matrix
    integer :: dimensions(2)      ! Matrix size (rows and columns)

    ! Size of the matrix
    dimensions = [100, 100]

    ! Matrix initialization
    call randInit(A, dimensions)

    ! Trace of the matrix (square matrix only)
    x = .Tr. A

    ! Adjoint matrix
    C = .Adj. A

    ! Writing the matrix on a text file : 
    call CMatDumpTXT(C, 'matrix_output.txt')
    print *, "The adjoint matrix has been written to 'matrix_output.txt'."

end program main
