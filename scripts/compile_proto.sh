echo "Compiling protobuf messages"
./protoc --python_out=. ./proto/*.proto
