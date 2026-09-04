using TestItemRunner
using MriReconstructionToolbox

# Run all tests. Example filtered runs from the package root:
#
#   julia --project=test -e '
#       using TestItemRunner
#       TestItemRunner.run_tests(pwd(); filter = ti -> :encoding in ti.tags)
#   '
#
# Available tags: :encoding, :regularization, :minimizer,
#                 :reconstruction, :integration, :nfft,
#                 :quality, :aqua, :jet, :acquisition, :simulation,
#                 :components, :operators, :preprocessing,
#                 :acquisition_info, :analysis, :fourier, :sensitivity_maps

TestItemRunner.run_tests(pkgdir(MriReconstructionToolbox))
