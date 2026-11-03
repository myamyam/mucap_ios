def solution(citations):
    answer = 0
    m = max(citations)
    n=len(citations)
    temp=[]
    while m>=0:
        for i in range(n):
            if citations[i]>=m:
                temp.append(citations[i])
            else:
                continue
        if len(temp)>=m:
            answer=m
            break
        else:
            temp=[]
            m-=1
    return answer

citaions=[28,10,5,3,3]
print(solution(citaions))