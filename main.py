import math
# Реализуем для регресси, если нужна классификация, то немного меняем 13 строку


def knn(data,query,distance_metric,k):
    pairwise_distance = []  # здесь храним расстояние между нашим входным значением(query) и значенияеми из data

    for index, line in enumerate(data):
        distance = distance_metric(line[:-1],query) # считаем расстояние с помощью метрики,которую определил пользователь
        pairwise_distance.append((distance,index)) # добавляем расстояние и индекс значения,для которого это расстояние посчитино
    pairwise_distance= sorted(pairwise_distance) # cортируем в порядке возврастания расстояний
    k_nearest = pairwise_distance[:k] # выбираем к ближайших
    k_nearest_labels = [data[i][1] for distance,i in k_nearest] # для этих к ближайших получаем значения
    return round(sum(k_nearest_labels)/len(k_nearest_labels)) # возвращаем среднее между "голосовавшими"

def euclidean_distance(p1,p2): # функция,для нахождения евклидового расстояние между точками, корень из суммы квадратов расстояний
    sum = 0
    for i in range(len(p1)):
        sum+= math.pow(p1[i] - p2[i],2)
    return math.sqrt(sum)

#predicting weight having only height values
reg_data = [
    [65.75, 112.99],
    [71.52, 136.49],
    [69.40, 153.03],
    [68.22, 142.34],
    [67.79, 144.30],
    [68.70, 123.30],
    [69.80, 141.49],
    [70.01, 136.46],
    [67.90, 112.37],
    [66.49, 127.45],
]

reg_query = [75] # наш входной пример
prediction = knn(reg_data,reg_query,distance_metric=euclidean_distance,k = 7) # delaem prediction

print("Result with k = 7 is {0}".format(prediction))
loss = []
ks = [1,2,3,4,5,6,7] # poprobuem vibrat` bolee podhodyashee k
for kc in ks: # dlya kazhdogo k
    avg_error = 0
    for i in range(len(reg_data)): # по очереди выкидываем строку из нашего массива,считая ее обучающей,чтобы посмотреть ошибку
        test_q = [reg_data[i][0]]
        test_answer = reg_data[i][1]
        test_data = reg_data.copy()
        del test_data[i]
        n_prediction = knn(test_data,test_q,distance_metric=euclidean_distance,k=kc)
        avg_error+=round(abs(n_prediction-test_answer),2)/len(reg_data)
    loss.append((round(avg_error,2),kc))
new_k = sorted(loss)[0][1] # выбираем к с наименьшей ошибкой
new_prediction = knn(reg_data,reg_query,distance_metric=euclidean_distance,k =new_k)
print("Result with k = {0} is {1}".format(new_k,new_prediction))
