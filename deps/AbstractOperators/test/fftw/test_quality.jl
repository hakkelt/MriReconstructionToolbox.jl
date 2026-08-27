@testitem "Aqua" tags = [:quality, :fftw] begin
    using Aqua, AbstractOperators, FFTW, FFTWOperators
    Aqua.test_all(
        FFTWOperators;
        piracies = false, persistent_tasks = VERSION >= v"1.11"
    )
    Aqua.test_piracies(FFTWOperators; treat_as_own = [FFTW, AbstractOperators])
end
