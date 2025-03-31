# 编译器
CXX = g++

# 编译器选项
CXXFLAGS = -std=c++11 -O2 -g

# oneDNN库路径（如安装在其他路径，请修改）
DNNL_LIB = /home/wangpy/zjw/install/oneDNN/lib
DNNL_INCLUDE = /home/wangpy/zjw/install/oneDNN/include

# 自动获取所有 .cpp 文件
SRC = $(wildcard *.cpp)

# 输出的可执行文件列表（与 .cpp 同名，后缀为 .out）
OUT = $(patsubst %.cpp, %.out, $(SRC))

# 链接oneDNN所需的选项
LDFLAGS = -ldnnl -lpthread

# 默认目标：编译所有 .cpp 文件
all: $(OUT)

# 模式规则：每个 .cpp 生成对应的 .out
%.out: %.cpp
	$(CXX) $(CXXFLAGS) -I$(DNNL_INCLUDE) -L$(DNNL_LIB) $< -o $@ $(LDFLAGS)

# 清理所有生成的文件
clean:
	rm -f $(OUT)