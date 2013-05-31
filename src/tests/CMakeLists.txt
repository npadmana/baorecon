include_directories("${PROJECT_SOURCE_DIR}")


foreach (prog ReadParticle RandomParticle SlabDecompose CIC CIC2 CIC3 pot pot2 pot_radial_timing ghost Xi Pk deriv Mask Xi_shell Xi_shell_smooth Fake HoffmanRibak Params ReadWriteAscii)
add_executable(test${prog} test${prog}.cpp)
target_link_libraries(test${prog} Recon ${PETSC_LIBS} ${FFTW_LIBS} ${GSL_LIBS})
endforeach(prog)
