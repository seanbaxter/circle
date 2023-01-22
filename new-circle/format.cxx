// Supports arithmetic arguments.
static_assert("The answer is {}.".format(42) == "The answer is 42.");

// Supports formatted arithmetic arguments.
static_assert("This is {:04x}.".format(0xc001) == "This is c001.");

// Supports named arguments.
static_assert("My name is {name}.".format(name: "Sean") == "My name is Sean.");

// Automatically converts enums to enum names.
enum class command {
  READ, WRITE, READWRITE
};

static_assert("Command is {}.".format(command::WRITE) == "Command is WRITE.");