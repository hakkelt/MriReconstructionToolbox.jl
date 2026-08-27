@testitem "Aqua" tags = [:quality, :nfft] begin
    using Aqua, NFFTOperators
    Aqua.test_all(NFFTOperators; persistent_tasks = VERSION >= v"1.11")
end
