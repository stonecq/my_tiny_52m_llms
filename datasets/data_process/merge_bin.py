import os
import sys


def merge_bin_files(input_dir, output_filename="merged_pretrain_data.bin"):
    """
    将指定目录下的所有 .bin 文件合并为一个文件。
    """
    # 1. 获取所有 .bin 文件路径
    # 使用 os.walk 确保能遍历子目录（如果有的话）
    bin_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.bin') and file != output_filename:  # 排除掉可能已存在的输出文件
                bin_files.append(os.path.join(root, file))

    # 2. 排序文件（非常重要！）
    # 确保文件按名称排序，保证数据顺序的一致性
    bin_files.sort()

    if not bin_files:
        print(f"❌ 在目录 '{input_dir}' 中未找到任何 .bin 文件。")
        return

    print(f"🔍 找到 {len(bin_files)} 个文件待合并：")
    for f in bin_files:
        print(f"   - {f}")

    output_path = os.path.join(input_dir, output_filename)

    # 防止覆盖正在使用的文件（可选的安全检查）
    if os.path.exists(output_path):
        print(f"⚠️ 警告: 输出文件 '{output_filename}' 已存在。")
        # 这里选择直接覆盖，如果你希望手动确认，可以取消下面注释
        # input("按回车键继续覆盖，或 Ctrl+C 退出...")

    print(f"\n🚀 开始合并...")

    total_size = 0
    try:
        with open(output_path, 'wb') as outfile:  # 以二进制写入模式打开输出文件
            for infile_path in bin_files:
                # 获取文件大小用于进度显示
                file_size = os.path.getsize(infile_path)

                with open(infile_path, 'rb') as infile:
                    # 分块读取和写入，避免大文件占用过多内存
                    # 每次读取 10MB (10 * 1024 * 1024 字节)
                    chunk_size = 10 * 1024 * 1024
                    while True:
                        chunk = infile.read(chunk_size)
                        if not chunk:
                            break
                        outfile.write(chunk)

                total_size += file_size
                print(f"✅ 已合并: {os.path.basename(infile_path)} ({file_size / 1024 / 1024:.2f} MB)")

        print("\n" + "=" * 40)
        print(f"🎉 合并完成！")
        print(f"📂 输出文件: {output_path}")
        print(f"📏 总大小: {total_size / 1024 / 1024:.2f} MB")
        print("=" * 40)

    except Exception as e:
        print(f"\n❌ 合并过程中发生错误: {e}")
        # 如果出错，尝试删除可能生成的损坏文件
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"🧹 已删除损坏的输出文件: {output_path}")


if __name__ == "__main__":
    # 配置区域：修改这里的路径
    # 如果你的脚本和数据集在同一级，可以直接写目录名
    target_directory = "/llama/datasets/tiny_llms/pre_train"

    # 允许通过命令行参数传递路径，例如: python merge_data.py /path/to/data
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]

    if not os.path.isdir(target_directory):
        print(f"❌ 错误: 目录 '{target_directory}' 不存在！")
    else:
        merge_bin_files(target_directory)