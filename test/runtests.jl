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
#                 :quality, :jet, :acquisition, :simulation,
#                 :components

TestItemRunner.run_tests(pkgdir(MriReconstructionToolbox))
