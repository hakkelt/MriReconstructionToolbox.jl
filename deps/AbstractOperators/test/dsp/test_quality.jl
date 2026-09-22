@testitem "Aqua" tags = [:quality, :dsp] begin
    using DSPOperators, Aqua
    Aqua.test_all(DSPOperators, persistent_tasks = VERSION >= v"1.11")
end
