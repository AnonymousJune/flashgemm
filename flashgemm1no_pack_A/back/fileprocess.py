input_file_path = '/home/wangpy/zjw/TEST/flashgemm1no_pack_A/back/kernel_f32_no_pack_A'

output_file_path = '/home/wangpy/zjw/TEST/flashgemm1no_pack_A/back/kernel_f32_no_pack_A_finished'

with open(input_file_path, 'r') as input_file:
    with open(output_file_path, 'w') as output_file:
        for line in input_file:
            if line.isspace() or line == "":
                output_file.write(line)
            else: 
                line = line.replace("\t", "    ")
                parts = line.split("//")
                if len(parts)==1:
                    write_string = "\"" + line.rstrip().ljust(60) + "\\n\"\n"
                elif parts[0].isspace() or parts[0]=="":
                    write_string = "//" + parts[1]
                else:
                    write_string = "\"" + parts[0].rstrip().ljust(60) + "\\n\"" + "//" + parts[1]
                output_file.write(write_string)
print("finish!")
