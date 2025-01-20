! Assignment 2, Exercise 3 (file 1)
! Gabriel Amorosetti

module mod_matrix_c8

    implicit none

    ! Definition of a derived type for a double precision complex matrix
    type :: complex8_matrix 
        integer, dimension(2) :: size                  ! Size of the matrix (array of dimension 2 representing the number of rows and columns)
        complex*8, dimension(:,:), allocatable :: elem ! Elements of the matrix
    end type complex8_matrix

    ! Interface for the operator to compute the adjoint
    interface operator(.Adj.)

        module procedure CMatAdjoint

    end interface

    ! Interface for the operator to compute the trace
    interface operator(.Tr.)

        module procedure CMatTrace

    end interface


contains

    ! Complex matrix initialization
    subroutine randInit(cmx, dims)

        type(complex8_matrix), intent(out) :: cmx   ! Complex matrix to initialize
        integer, dimension(2), intent(in) :: dims   ! Size of the matrix (array of dimension 2 representing the number of rows and columns)
        real(8), allocatable :: real_part(:,:), imag_part(:,:) ! Real and imaginary parts of the complex element

        cmx%size = dims                             ! Assign the input dimensions to the complex matrix

        allocate(cmx%elem(dims(1), dims(2)))        ! Allocation of the matrix elements
        allocate(real_part(dims(1), dims(2)))       ! Allocation of the real parts
        allocate(imag_part(dims(1), dims(2)))       ! Allocation of the imaginary parts

        ! Initialization of the real and imaginary parts with random values (between 0 and 1)
        call random_seed()
        call random_number(real_part)
        call random_number(imag_part)

        ! Combining these parts into complex matrix elements and scaling values by 100.0
        cmx%elem = cmplx(real_part * 100.0, imag_part * 100.0, kind=8)

        ! Deallocation of the real and imaginary parts
        deallocate(real_part, imag_part)
    
    end subroutine randInit


    ! Function to compute the adjoint matrix
    function CMatAdjoint(cmx) result(cmxadj)

        type(complex8_matrix), intent(in) :: cmx ! Input matrix
        type(complex8_matrix) :: cmxadj          ! Output matrix

        ! Setting size of the adjoint matrix (transpose of input dimensions)
        cmxadj%size(1) = cmx%size(2)
        cmxadj%size(2) = cmx%size(1)

        allocate(cmxadj%elem(cmxadj%size(1), cmxadj%size(2))) ! Allocating adjoint matrix elements
        cmxadj%elem = conjg(transpose(cmx%elem))              ! Computing the conjugate transpose of the elements

    end function

    ! Function to compute the trace of the matrix
    function CMatTrace(cmx) result(tr)

        type(complex8_matrix), intent(in) :: cmx ! Input matrix
        complex*8 :: tr                          ! Output (trace)
        integer :: ii

        tr = complex(0.0, 0.0)  ! Initialization to zero

        ! Summing up diagonal elements
        do ii = 1, cmx%size(1)
            tr = tr + cmx%elem(ii, ii)
        end do

    end function

    ! Subroutine to write the matrix on a text file
    subroutine CMatDumpTXT(cmx, file_name)

        type(complex8_matrix), intent(in) :: cmx    ! Input matrix
        character(len=*), intent(in) :: file_name   ! Output file name
        integer :: i, j
    
        open(unit=10, file=file_name, status='replace', action='write')
        write(10, *) 'Matrix Dimensions: ', cmx%size(1), 'x', cmx%size(2)   ! Writing matrix size

        ! Writing each row of the matrix
        do i = 1, cmx%size(1)
            ! Adjust format to output each element in a row, with a space between elements
            write(10, '(100F10.4)') (cmx%elem(i, j), j = 1, cmx%size(2))    ! Adjusting format for each element
        end do

        close(10)

    end subroutine CMatDumpTXT

end module mod_matrix_c8
