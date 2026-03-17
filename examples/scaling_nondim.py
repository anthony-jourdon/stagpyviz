import stagpyviz as spv

def print_factors(scaling_factors:dict[str, spv.Scaling]):
  for key in scaling_factors:
    print(f"{scaling_factors[key]}")
  return

def get_value(fname:str=None) -> float:
  val = input(f"Enter {fname} value: ")
  try: 
    val = float(val)
  except:
    print(f"Invalid value entered, expecting a number, found: {val}. Retry.")
    val = get_value(fname)
  return val

def get_unit(fname:str=None) -> str|None:
  val = input(f"Enter {fname} unit (return for no unit): ")
  if val == "":
    return None
  return val

def get_field_name(fields_list: list) -> str:
  val = input("Enter a Field name: ")
  if val == "":
    print("No field name entered, exiting.")
    exit(0)
  if val not in fields_list:
    print(f"Invalid field name entered, expecting one of: {fields_list}, found: {val}. Retry.")
    val = get_field_name(fields_list)
  return val

def main():
  Ra = get_value("Rayleigh number")
  scaling_factors = spv.scaling_factors(Ra=Ra)

  mode = input("Enter mode (dim/adim/print): ")
  if mode not in ["dim","adim","print"]:
    print(f"Invalid mode entered, expecting 'dim', 'adim' or 'print', found: {mode}")
    exit(1)

  if mode == "print":
    print_factors(scaling_factors)
    exit(0)

  field_list = list(scaling_factors.keys())
  while True:
    fname = get_field_name(field_list)
    val   = get_value(fname)
    unit  = get_unit(fname)
    if mode == "adim":
      if unit is not None:
        val = scaling_factors[fname].to_base(val, unit)
      value = scaling_factors[fname].a_dim(val)
      print(f"Adimensional value for {fname}: {value}")
    else:
      val = scaling_factors[fname].dim(val)
      if unit is not None:
        val = scaling_factors[fname].to(val, unit)
      print(f"Dimensional value for {fname}: {val} {unit}")

  return

if __name__ == "__main__":
  main()