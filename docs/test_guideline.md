## Write tests:
* We use [GoogleTest](https://google.github.io/googletest/) as the test framework.
* Refer to [tests/framework](../tests/framework/) for a sample of how to write unit tests.
* Please write test code as much as possible.

## How to run tests
* The tests are built along with Physika. After the build completes, run 'ctest' command in build directory to run all the tests. Example:
```
mkdir -p build
cd build
// cmake && build commands
//...
ctest  //run all the tests
```
* To run a specific test, enter the path of the test executive and run it in command line. Example:
```
mkdir -p build
cd build
// cmake && build commands
//...
cd ./tests/framework
./framework_test   //run tests of framework only
```