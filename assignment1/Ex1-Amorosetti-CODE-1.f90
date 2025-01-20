! Assignment 1, Exercise 1
! Gabriel Amorosetti

program Setup
    implicit none   ! To enforce explicit variable declaration
    integer :: a,b
    real :: x, r

    ! Simple calculation as a test job
    x = 3.2
    a = 3
    b = 4
    r = b*x - a

    print *, "Hello world, here is a simple calculation : 4*3.2 - 3 =",r

end program Setup

