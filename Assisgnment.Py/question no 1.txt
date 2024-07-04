ask=int(input("Enter the age:"))


if ask<18:
  print("you are minor")

elif ask>18 and ask<60:
  print("you are adult")
elif ask>60 and ask<110:
  print("you are senior citizen")

else:
  print("you are dead")