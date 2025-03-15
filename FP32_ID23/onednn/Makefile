# 编译器
CXX = g++

# 编译器选项
CXXFLAGS = -std=c++11 -O2 -g

# oneDNN库路径（如安装在其他路径，请修改）
DNNL_LIB = /home/wangpy/zjw/install/oneDNN_debug/lib
DNNL_INCLUDE = /home/wangpy/zjw/install/oneDNN_debug/include

# 源文件
SRC = matmul.cpp

# 输出的可执行文件名称
OUT = matmul

# 链接oneDNN所需的选项
LDFLAGS = -ldnnl -lpthread

# 编译规则
all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) -I$(DNNL_INCLUDE) -L$(DNNL_LIB) $(SRC) -o $(OUT) $(LDFLAGS)

# 清理编译生成的文件
clean:
	rm -f $(OUT)