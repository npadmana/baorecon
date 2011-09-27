add_definitions(-DTIXML_USE_TICPP)
ADD_LIBRARY(ticpp SHARED ticpp.h ticpprc.h tinystr.h tinyxml.h ticpp.cpp tinystr.cpp tinyxml.cpp tinyxmlerror.cpp tinyxmlparser.cpp)

# examples are below
#add_executable(example example.cpp)
#target_link_libraries(example ticpp)

