! Assignment 2, Exercise 1
! Gabriel Amorosetti

module debugger
    
    implicit none

contains 

    subroutine checkpoint(debug, verb, msg, var)

        logical :: debug                    ! Logical variable, can be .true. or .false.
        integer, optional :: verb           ! Variable for the level of verbosity wanted
        character(len=*), optional :: msg   ! String of variable lenght 
        integer, optional :: var            ! Numerical variable 
        ! Optional variables don't need to be used when the subroutine will be called

        integer :: verb_value  ! Temporary variable to store verb value


        if (.not. present(verb)) then 
            verb_value = 2      ! Set default value for verb to 2, if not provided when called
        else
            verb_value = verb   ! If provided, use the one chosen by user
        end if
        
        if (debug) then         ! If debug variable is set to .true. when the subroutine is called

            if (verb >= 0) then ! First level of verbosity : only a message to signal the subroutine is called is printed
                print *, " "
                print *, "Checkpoint called"
                print *, " "
            end if

            if (verb >= 1) then ! Second level of verbosity : on top of the previous print, print a message set by user
                print *, msg
            end if

            if (verb >= 2) then ! Third level of verbosity : on top of the previous prints, print a variable set by user
                print *, var    ! var needs to be integer !
            end if

        else    ! If debug variable is not set to .true. when the subroutine is called, nothing happens
            ! no print

        end if

    end subroutine checkpoint

end module debugger


