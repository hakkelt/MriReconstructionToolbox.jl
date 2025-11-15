macro step(step_name, config, expr)
    quote
        if $(esc(config)).verbose
            task = @spawn @timed $(esc(expr))
            is_stepname_printed = Threads.Atomic{Bool}(false)
            step_name = $(esc(step_name))
            timer = Timer(1) do _
                if !is_stepname_printed[]
                    is_stepname_printed[] = true
                    $(esc(config)).printfunc("Starting ", lowercasefirst(step_name), "...")
                end
            end
            stats = fetch(task)
            if is_stepname_printed[]
                $(esc(config)).printfunc("Finished ", lowercasefirst(step_name), " in ", format_stats(stats))
            else
                is_stepname_printed[] = true
                close(timer)
                $(esc(config)).printfunc(uppercasefirst(step_name), ": ", format_stats(stats))
            end
        else
            $(esc(expr))
        end
    end
end

macro printing_step(step_name, config, expr)
    quote
        if $(esc(config)).verbose
            step_name = $(esc(step_name))
            $(esc(config)).printfunc("Starting ", lowercasefirst(step_name), "...")
            stats = @timed $(esc(expr))
            $(esc(config)).printfunc("Finished ", lowercasefirst(step_name), " in ", format_stats(stats))
        else
            $(esc(expr))
        end
    end
end

function format_stats(stats)
    io = IOBuffer()
    time_print(io, stats)
    return String(take!(io))
end

function pretty_bytes(bytes::Int)
    number, unit = prettyprint_getunits(bytes, length(_mem_units), Int64(1024))
    if unit == 1
        return "$(Int(number)) $(_mem_units[unit])"
    else
        return "$(round(number, digits=2)) $(_mem_units[unit])"
    end
end

const _mem_units = ["byte", "KiB", "MiB", "GiB", "TiB", "PiB"]
const _cnt_units = ["", " k", " M", " G", " T", " P"]
function prettyprint_getunits(value, numunits, factor)
    if value == 0 || value == 1
        return (value, 1)
    end
    unit = ceil(Int, log(value) / log(factor))
    unit = min(numunits, unit)
    number = value/factor^(unit-1)
    return number, unit
end

function time_print(io::IO, stats)
    print(io, format_time(stats.time))
    if VERSION < v"1.11"
        stats = (lock_conflicts=0, compile_time=0, recompile_time=0, stats...)
    end
    parens = stats.bytes != 0 || stats.bytes != 0 || stats.gctime > 0 || stats.lock_conflicts > 0 || stats.compile_time > 0
    parens && print(io, " (")
    if stats.bytes != 0 || stats.bytes != 0
        allocs, ma = prettyprint_getunits(stats.bytes, length(_cnt_units), Int64(1000))
        if ma == 1
            print(io, Int(allocs), _cnt_units[ma], allocs==1 ? " allocation: " : " allocations: ")
        else
            print(io, round(allocs, digits=2), _cnt_units[ma], " allocations: ")
        end
        print(io, Base.format_bytes(stats.bytes))
    end
    if stats.gctime > 0
        if stats.bytes != 0 || stats.allocs != 0
            print(io, ", ")
        end
        print(io, round(100*stats.gctime/stats.time, digits=2), "% gc time")
    end
    if stats.lock_conflicts > 0
        if stats.bytes != 0 || stats.allocs != 0 || stats.gctime > 0
            print(io, ", ")
        end
        plural = stats.lock_conflicts == 1 ? "" : "s"
        print(io, stats.lock_conflicts, " lock conflict$plural")
    end
    if stats.compile_time > 0
        if stats.bytes != 0 || stats.allocs != 0 || stats.gctime > 0 || stats.lock_conflicts > 0
            print(io, ", ")
        end
        print(io, round(100*stats.compile_time/stats.time, digits=2), "% compilation time")
    end
    if stats.recompile_time > 0
        perc = Float64(100 * stats.recompile_time / stats.compile_time)
        # use "<1" to avoid the confusing UX of reporting 0% when it's >0%
        print(io, ": ", perc < 1 ? "<1" : round(perc, digits=0), "% of which was recompilation")
    end
    parens && print(io, ")")
end

function format_time(t::Float64)
    if t < 1e-3
        return "$(round(t*1e6, digits=2)) µs"
    elseif t < 1
        return "$(round(t*1e3, digits=2)) ms"
    elseif t < 60
        return "$(round(t, digits=2)) s"
    elseif t < 3600
        m = floor(Int, t / 60)
        s = round(t - m * 60, digits=2)
        return "$m min $s s"
    else
        h = floor(Int, t / 3600)
        m = floor(Int, (t - h * 3600) / 60)
        s = round(t - h * 3600 - m * 60, digits=2)
        return "$h h $m min $s s"
    end
end