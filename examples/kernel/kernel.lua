print("lua locked and loaded")

function printf(...)
  io.write(string.format(...))
end

type_sizes = {
  char = 1,
  short = 2,
  int = 4,
  long = 8,
  float = 4,
  double = 8
}

function is_float(type)
  return "float" == type or "double" == type
end

function kernel_params(key)
  -- This function has your kernel's special sauce. It runs each time the
  -- kernel function template is instantiated, and key has fields
  --   int sm
  --   string type
  -- describing the template parameters.

  -- This file is not distributed with the resulting executable. It and the
  -- Lua interpreter are used only at compile-time. However, the luacir.hxx
  -- Circle/Lua bindings work at both compile-time (inside the interpreter)
  -- and runtime.

  printf("  **Lua gets key { %d, %s }\n", key.sm, key.type)
  params = { }
  params.flags = { }
  if is_float(key.type) and key.sm > 52 then
    params.flags[1] = "fast_math"
  end

  if key.type == "short" then
    params.bytes_per_lane = 8
    if key.sm < 50 then
      params.lanes_per_thread = 2
    else
      params.lanes_per_thread = 4
    end

  elseif key.type == "float" then
    params.bytes_per_lane = 16
    if key.sm < 50 then
      params.lanes_per_thread = 4
    else
      params.lanes_per_thread = 8
    end
    params.flags[#params.flags + 1] = "ftz"

  else
    params.bytes_per_lane = 24
    params.lanes_per_thread = 32 // type_sizes[key.type]
    params.flags[#params.flags + 1] = "ldg"

  end
  return params
end