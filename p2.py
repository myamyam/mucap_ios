import math as m

def solution(r1, r2):
    answer = 0
    for x in range(1,r2):
        for y in range(1,r2):
            if m.sqrt(x**2+y**2)>=r1 and m.sqrt(x**2+y**2)<=r2:
                answer+=4
    answer+=4*(r2-r1+1)
    return answer

# for statement 두개 시간제한 걸림