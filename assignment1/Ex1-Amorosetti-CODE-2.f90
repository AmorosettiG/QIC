! Assignment 1, Exercise 2
! Gabriel Amorosetti

program NumberPrecision

    ! implicit none ! Not used here to use 1.0E32, 1.0E21, etc..., without explicit variable declaration

    integer(2) :: a_2, b_2, c_2 ! Represents 16-bit integers
    integer(4) :: a_4, b_4, c_4 ! Represents 32-bit integers

    real(4) :: x_1, y_1, z_1    ! Single precision real numbers (32-bit)
    real(8) :: x_2, y_2, z_2    ! Double precision real numbers (64-bit)


    !--------------------------------------------
    !--------------------------------------------
    
    !!!! Sum 2 000 000 and 1 with INTEGER*2 !!!!

    a_2 = 2000000   ! Arithmetic overflow, 2 000 000 is too big for a 16-bit integer (only goes from -32 768 to 32 767)
    b_2 = 1
    c_2 = a_2 + b_2

    ! c_2 should be 2 000 001, but because of the overflow, the code can't be compiled
    ! We get a wrong result when forcing the compilation with '-fno-range-check'

    !!!! Sum 2 000 000 and 1 with INTEGER*4 !!!!

    a_4 = 2000000
    b_4 = 1
    c_4 = a_4 + b_4
    
    print *, "Sum 2 000 000 and 1 with INTEGER*2 : ", c_2
    print *, "Sum 2 000 000 and 1 with INTEGER*4 : ", c_4

    !--------------------------------------------
    !--------------------------------------------

    !!! Sum of pi*10^32 and sqrt(2)*10^21 with single precision !!!

    x_1 = acos(-1.0_4)*1.0E32 ! We use arccosine(-1) to get pi, -1.0_4 is -1.0 with single precision
    y_1 = sqrt(2.0_4)*1.0E21  ! 1.0E21 is 10^21, E is used for single precision
    z_1 = x_1 + y_1

    !!! Sum of pi*10^32 and sqrt(2)*10^21 with double precision !!!

    x_2 = acos(-1.0_8)*1.0D32 ! We use arccosine(-1) to get pi, -1.0_8 is -1.0 with double precision
    y_2 = sqrt(2.0_8)*1.0D21  ! 1.0D21 is 10^21, D is used for double precision
    z_2 = x_2 + y_2

    print *, "Sum of pi*10^32 and sqrt(2)*10^21 with single precision : ", z_1
    print *, "Sum of pi*10^32 and sqrt(2)*10^21 with double precision : ", z_2

end program NumberPrecision
