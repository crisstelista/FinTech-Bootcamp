def isPalindrome(s):
    return s == s[::-1]

print(isPalindrome("mother"))
print(isPalindrome("momom"))