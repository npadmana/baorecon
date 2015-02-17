include_directories("${PROJECT_SOURCE_DIR}")


foreach (prog recon_PBC_real recon_shell_real recon_shell_zpp recon_shell_radial recon_lasdamas_real recon_lasdamas_zspace recon_lasdamas_real_weighted 
        recon_lasdamas_zspace_weighted recon_lasdamas_zspace_weighted_norsd)
add_executable(${prog} ${prog}.cpp)
target_link_libraries(${prog} Recon ${PETSC_LIBS} ${FFTW_LIBS} ${GSL_LIBS} ticpp)
endforeach(prog)

