import random
import string
from typing import Optional


class NameGenerator:
    """英文人名生成器"""
    
    # 英文名字（First Names）
    FIRST_NAMES = [
        "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Christopher",
        "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
        "Kenneth", "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan",
        "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon",
        "Benjamin", "Samuel", "Frank", "Gregory", "Raymond", "Alexander", "Patrick", "Jack", "Dennis", "Jerry",
        "Tyler", "Aaron", "Jose", "Adam", "Nathan", "Henry", "Douglas", "Zachary", "Peter", "Kyle",
        "Walter", "Ethan", "Jeremy", "Harold", "Seth", "Christian", "Mason", "Austin", "Gerald", "Roger",
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
        "Nancy", "Lisa", "Betty", "Helen", "Sandra", "Donna", "Carol", "Ruth", "Sharon", "Michelle",
        "Laura", "Emily", "Kimberly", "Deborah", "Dorothy", "Lisa", "Nancy", "Karen", "Betty", "Helen",
        "Sandra", "Donna", "Carol", "Ruth", "Sharon", "Michelle", "Laura", "Emily", "Kimberly", "Deborah",
        "Dorothy", "Lisa", "Nancy", "Karen", "Betty", "Helen", "Sandra", "Donna", "Carol", "Ruth"
    ]
    
    # 英文姓氏（Last Names）
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
        "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
        "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
        "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker", "Cruz", "Edwards", "Collins", "Reyes",
        "Stewart", "Morris", "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper",
        "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
        "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
        "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long", "Ross", "Foster", "Jimenez"
    ]
    
    @classmethod
    def generate_english_name(cls, first_name: Optional[str] = None, last_name: Optional[str] = None) -> str:
        """生成英文人名
        
        Args:
            first_name: 可选的名字，如果不提供则随机选择
            last_name: 可选的姓氏，如果不提供则随机选择
            
        Returns:
            生成的英文人名
        """
        if first_name is None:
            first_name = random.choice(cls.FIRST_NAMES)
        if last_name is None:
            last_name = random.choice(cls.LAST_NAMES)
        
        return f"{first_name} {last_name}"
    
    @classmethod
    def generate_random_string(cls, length: int = 4, include_digits: bool = True) -> str:
        """生成随机字符串
        
        Args:
            length: 字符串长度
            include_digits: 是否包含数字
            
        Returns:
            生成的随机字符串
        """
        chars = string.ascii_letters
        if include_digits:
            chars += string.digits
        
        return ''.join(random.choice(chars) for _ in range(length))
    
    @classmethod
    def generate_name(cls, excluded_names: list[str] = None, max_attempts: int = 100) -> str:
        """生成唯一的英文人名
        
        Args:
            excluded_names: 需要排除的名字列表，默认为空列表
            max_attempts: 最大尝试次数
            
        Returns:
            生成的唯一英文人名
        """
        if excluded_names is None:
            excluded_names = []
        
        for _ in range(max_attempts):
            name = cls.generate_english_name()
            if name not in excluded_names:
                return name
        
        # 如果无法生成唯一名字，添加随机字符串
        base_name = cls.generate_english_name()
        random_suffix = cls.generate_random_string(4, True)
        return f"{base_name}_{random_suffix}"


# 便捷函数
def generate_name(excluded_names: list[str] = None) -> str:
    """生成唯一英文人名的便捷函数
    
    Args:
        excluded_names: 需要排除的名字列表，默认为空列表
        
    Returns:
        生成的唯一英文人名
    """
    return NameGenerator.generate_name(excluded_names) 
