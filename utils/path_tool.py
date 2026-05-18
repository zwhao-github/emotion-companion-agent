"""
为整个工程提供统一的绝对路径
"""
from pathlib import Path


def get_project_root() -> str:
    """
    获取工程所在的根目录
    :return: 字符串根目录
    """
    # 当前文件的绝对路径
    current_file = Path(__file__).absolute()
    # 获取工程的根目录，先获取文件所在的文件夹绝对路径
    current_dir = current_file.parent
    # 获取工程根目录
    project_root = current_dir.parent

    return str(project_root)


def get_abs_path(relative_path: str) -> str:
    """
    传递相对路径，得到绝对路径
    :param relative_path: 相对领
    :return: 绝对路径
    """
    project_root = get_project_root()
    return str(Path(project_root) / relative_path)


if __name__ == '__main__':
    print(get_abs_path("config/rag.yml"))
