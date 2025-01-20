! Assignment 2, Exercise 2
! Gabriel Amorosetti

! ============================================================================
! DOCUMENTATION

! ----------------------------------------------------------------------------

! This program implements the matrix-matrix multiplication with differents methods, in order to compare execution times

! Several methods were tested : 

! - Standard A*B =C matrix multiplication following the usual formula A*B = C ---> for k = 1 to n : C(i,j) = sum(A(i,k) * B(k,j) 
! - Column-wise matrix multiplication, adapting the iterations on the standard formula to access and fill the matrices only by column
! - Parallelized version of the standard matrix multiplication, with the omp_lib library, to take advantages of multi-cores CPU
! - Parallelized version of the column-wise matrix multiplication, with the omp_lib library, to take advantages of multi-cores CPU
! - Intrinsic Fortran function to multiply two matrices C = matmul(A,B)

! ----------------------------------------------------------------------------

! Furthermore, the program was compiled with different optimization flags listed here, again to compare execution times :

! Optimization flags tested  with 'gfortran -fopenmp -o a1_ex3.exe a1_ex3.f90': 

! No optimization flags (only -fopenmp to enable parallel computation)
! -O1
! -O2
! -O3
! -Ofast 
! -funroll-loops
! -Ofast -march=native -mtune=native -mavx512f -mavx512dq -ftree-vectorize -ftree-loop-vectorize -funroll-loops

! Some notes about these optimization flags : 

! -O flags here are sorted from the least to most agressive optimization performed by the compiler
! The most agressives -O flags already includes flags like -fstack-arrays -ffast-math
! -funroll-loops transforms a loop by duplicating its body multiple times
! -march=native -mtune=native optimize the code specifically for the machine it’s being compiled on
! -mavx512f -mavx512dq enable specific sets of AVX-512 instructions on CPUs that support them
! -ftree-vectorize -ftree-loop-vectorize optimizations specifically focused on vectorizing operations

! -fbounds-check : 
!  Used to debug, adds checks to ensure that any access to array elements is within the defined bounds of the array


!-----------------------------------------------------------------
! My CPU : Ryzen 7 8840HS, 8 cores, 16 Threads
! Computer running on Windows 11, set on performance mode both in System settings and Lenovo Vantage app settings 
! (found it affects execution time if not set properly)

!-----------------------------------------------------------------

program TestingPerformance

    use omp_lib     ! OpenMP module to enable parallel programming
    use debugger    ! debugger module (Exercise 1 of the 2nd assignment) to call the checkpoint subroutine
    
    implicit none
    integer :: n, threads  
    real(8) :: time, start_time, end_time, time_omp, start_time_omp, end_time_omp
    integer, allocatable :: A(:,:), B(:,:), C(:,:)

    ! n is the size of the 2 square matrices
    ! threads is the number of threads we want to set for the parallel computation
    ! time, start_time, end_time, time_omp, start_time_omp, end_time_omp are the variables uses to measure the execution time
    ! A(:,:), B(:,:), C(:,:) : Matrices, multiplication of A and B, giving C, where the size is decided when calling the subroutine

    !-----------------------------------------------------------------

    ! Multiplication of two square matrices of size n
 
    print *,"########################################################"
    print *, " "
    print *,"Multiplication of two square matrices of size n"
    print *, " "
    print *, "Input n and push 'Enter'"
    print *, " "
    read *, n   ! Prompt n to the user to ask the size of the matrices to multiply


    ! Pre-condition to check that n (Matrices size) is superior or equal to 1
    if (n < 1) then
        print *, "Error, n must be superior or equal to 1"
        stop     ! Error handling to stop the program if the input is incorrect
    end if



    allocate(A(n, n), B(n,n)) ! Memory allocation of the A and matrices of size n

    call matrix_init(n, A ,B) ! Call the subroutine to initialize A and B 
    
    !-----------------------------------------------------------------
    ! prints used to check the content of A and B (only for small matrices)

    ! print *, "-------------"

    ! do i = 1, n
    !     print *, A(i,:)
    ! end do

    ! print *, "-------------"

    ! do i = 1, n
    !     print *, B(i,:)
    ! end do
    !-----------------------------------------------------------------

    allocate(C(n,n))        ! Memory allocation of the matrix C, result of the multiplication

    !-----------------------------------------------------------------

    ! Execution time tracking :

    ! CPU_TIME accumulates the individual CPU times for each thread, 
    ! which results in a higher overall time when multiple cores are working simultaneously

    ! So for the parallelized implemations of the multiplication,
    ! we use time from the omp module to get the "real" time  

    print *,"----------------------------------------------------"
    print *,"Square matrices of size n = ", n
    print *,"----------------------------------------------------"

    call CPU_TIME(start_time)
    call matrix_multiplication_byrow(n, A,B,C)      ! Standard matrices multiplication
    call CPU_TIME(end_time)


    call checkpoint(.true., 1, "Multiplication done, standard")

    time = end_time - start_time
    print *,"CPU time", time, "seconds"

    call check_post_conditions(C,n)

    print *,"----------------------------------------------------"
    
    call CPU_TIME(start_time)
    call matrix_multiplication_bycolumn(n, A,B,C)   ! Column-wise matrices multiplication
    call CPU_TIME(end_time)

    call checkpoint(.true., 1, "Multiplication done, column-wise")

    time = end_time - start_time
    print *,"CPU time", time, "seconds"

    call check_post_conditions(C,n)

    print *,"----------------------------------------------------"

    call CPU_TIME(start_time)
    C = matmul(A,B)                                 ! Built-in Fortran function for matrices multiplication
    call CPU_TIME(end_time)

    time = end_time - start_time

    call checkpoint(.true., 1, "Multiplication done with built-in function")
    !print *, "Multiplication done with built-in function"

    print *,"CPU time", time, "seconds"

    call check_post_conditions(C,n)

    print *,"----------------------------------------------------"

    threads = 14
    call omp_set_num_threads(min(threads, omp_get_max_threads()))

    ! print *, "Threads : ", threads
    call checkpoint(.true., 2, "Number of threads :", threads)

    print *,"----------------------------------------------------"

    start_time_omp = omp_get_wtime()
    call matrix_multiplication_byrow_parallel(n, A,B,C)     ! Standard matrices multiplication, parallelized
    end_time_omp = omp_get_wtime()

    call checkpoint(.true., 1, "Multiplication done, standard, parallelized")

    time_omp = end_time_omp - start_time_omp
    print *,"Real time : ", time_omp, "seconds"

    call check_post_conditions(C,n)

    print *,"----------------------------------------------------"

    start_time_omp = omp_get_wtime()
    call matrix_multiplication_bycolumn_parallel(n, A,B,C)  ! Column-wise matrices multiplication, parallelized
    end_time_omp = omp_get_wtime()

    call checkpoint(.true., 1, "Multiplication done, column-wise, parallelized")

    time_omp = end_time_omp - start_time_omp
    print *,"Real time : ", time_omp, "seconds"

    call check_post_conditions(C,n)

    print *,"----------------------------------------------------"


    !-----------------------------------------------------------------
    ! print used to check the content of C (only for small matrices)

    ! print *, "-------------"
    ! do i = 1, n
    !     print *, C(i,:)
    ! end do
    !-----------------------------------------------------------------

    deallocate(A, B, C)     ! Deallocation of A, B and C matrices from the memory

    print *, " "
    print *,"########################################################"

contains

    subroutine matrix_init(m, mat1, mat2)

        ! Subroutine created to initialize the two matrices to multiply


        integer, intent(in) :: m    ! intent(in) : means that the variable is only used for reading
        integer, intent(out) :: mat1(m, m), mat2(m, m) ! intent(out) : output only
        integer :: i, j

        ! m : size of the 2 square matrices
        ! mat1(m, m), mat2(m, m) : matrices to initialize
        ! i, j, used to iterate on the elements of the matrices to give them a value

        do i = 1, m
            do j = 1, m
                mat1(i, j) = i * (j**3)
            end do
        end do

        call checkpoint(.true., 1, "First matrix initialized")

        do i = 1, m
            do j = 1, m
                mat2(i, j) = (i**2) * j
            end do
        end do

        call checkpoint(.true., 1, "Second matrix initialized")

        ! For analysis consistency, the two matrices will always be initialized the same way : 
        ! For the first one, A, A_ij = i * (j^3)
        ! For the second one, B, B_ij = (i^2) * j

    end subroutine matrix_init

    subroutine matrix_multiplication_byrow(m, mat1, mat2, mat3)

        ! Subroutine implementing the standard formula for matrices multiplication A*B =C : 
        ! for k = 1 to n : 
        !   C(i,j) = sum(A(i,k) * B(k,j) 

        integer, intent(in) :: m                        ! Size of the matrices
        integer, intent(in) :: mat1(m, m), mat2(m, m)   ! Input matrices, A and B
        integer, intent(out) :: mat3(m, m)              ! Result of the multiplication, C
        integer :: i, j, k                              ! To iterate on the elements of C

        mat3 = 0.0  ! Initialization of C
                
        do i = 1, m
            do j = 1, m

                do k = 1, m
                    mat3(i, j) = mat3(i, j) + mat1(i, k)*mat2(k, j)
                end do

            end do
        end do

        !print *, "Multiplication done, standard"

    end subroutine matrix_multiplication_byrow

    subroutine matrix_multiplication_byrow_parallel(m, mat1, mat2, mat3)

        ! Exactly the same subroutine as the previous one, but with OpenMP directives,
        ! in order to use all the cores of the CPU

        integer, intent(in) :: m
        integer, intent(in) :: mat1(m, m), mat2(m, m)
        integer, intent(out) :: mat3(m, m)
        integer :: i, j, k

        mat3 = 0.0
        
        ! OpenMP directive
        !$omp parallel do collapse(2) private(i, j, k) shared(mat1, mat2, mat3) 
        
        do i = 1, m
            do j = 1, m

                do k = 1, m
                    mat3(i, j) = mat3(i, j) + mat1(i, k)*mat2(k, j)
                end do

            end do
        end do

        ! OpenMP directive
        !$omp end parallel do

        !print *, "Multiplication done, standard, parallelized"

    end subroutine matrix_multiplication_byrow_parallel

    subroutine matrix_multiplication_bycolumn(m, mat1, mat2, mat3)

        ! Subroutine implementing the A*B multiplication, accessing and filling the matrices only by column

        integer, intent(in) :: m
        integer, intent(in) :: mat1(m, m), mat2(m, m)
        integer, intent(out) :: mat3(m, m)
        integer :: i, j, k

        mat3 = 0.0
        
        do j = 1, m
            do k = 1, m

                do i = 1, m
                    mat3(i, j) = mat3(i, j) + mat1(i, k)*mat2(k, j)
                end do

            end do
        end do
        !print *, "Multiplication done, column-wise"

    end subroutine matrix_multiplication_bycolumn

    subroutine matrix_multiplication_bycolumn_parallel(m, mat1, mat2, mat3)

        ! Exactly the same subroutine as the previous one, but with OpenMP directives,
        ! in order to use all the cores of the CPU

        integer, intent(in) :: m
        integer, intent(in) :: mat1(m, m), mat2(m, m)
        integer, intent(out) :: mat3(m, m)
        integer :: i, j, k

        mat3 = 0.0
        
        ! OpenMP directive
        !$omp parallel do collapse(2) private(i, j, k) shared(mat1, mat2, mat3)

        do j = 1, m
            do k = 1, m

                do i = 1, m
                    mat3(i, j) = mat3(i, j) + mat1(i, k)*mat2(k, j)
                end do

            end do
        end do

        ! OpenMP directive
        !$omp end parallel do

        !print *, "Multiplication done, column-wise, parallelized"

    end subroutine matrix_multiplication_bycolumn_parallel

    subroutine check_post_conditions(Z, m)
        integer, intent(in) :: Z(:,:) ! Output of the multiplication
        integer, intent(in) :: m      ! Input matrices size
        ! Subroutine used to check that the output matrix has good dimensions
        if (size(Z, 1) /= m .or. size(Z, 2) /= m) then
            print *, " "
            print *, "Error: Result matrix has incorrect dimensions."
        else
            print *, " "
            print *, "Post-condition: Result matrix has correct dimensions."
        end if
    end subroutine

end program TestingPerformance
