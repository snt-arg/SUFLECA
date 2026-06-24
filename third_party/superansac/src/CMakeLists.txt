# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find Boost headers used by MAGSAC scoring.
find_package(Boost REQUIRED)

# Core shared library
add_library(SupeRANSAC SHARED
  superansac.cpp
)

# Give a unique, stable SONAME to avoid stale copies being picked at runtime
set_target_properties(SupeRANSAC PROPERTIES
  OUTPUT_NAME superansac_core
  SOVERSION 1
  VERSION   1.0.0
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN NO
)

# Public includes
target_include_directories(SupeRANSAC PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

# Link deps
target_link_libraries(SupeRANSAC
  Eigen3::Eigen
)

# Install next to the Python module
install(TARGETS SupeRANSAC
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .
)
