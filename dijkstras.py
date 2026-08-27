# 建立圖 (圖的相鄰表)
graph = {}
graph["start"] = {}       # 初始化起始節點
graph["start"]["a"] = 6   # 起始節點到 a 的邊的權重為 6
graph["start"]["b"] = 2   # 起始節點到 b 的邊的權重為 2

graph["a"] = {}           # 初始化節點 a
graph["a"]["fin"] = 1     # a 到終點 fin 的邊的權重為 1

graph["b"] = {}           # 初始化節點 b
graph["b"]["a"] = 3       # b 到 a 的邊的權重為 3
graph["b"]["fin"] = 5     # b 到終點 fin 的邊的權重為 5

graph["fin"] = {}         # 初始化終點 fin

# 成本表
infinity = float("inf")   # 設定無窮大
costs = {}
costs["a"] = 6            # 起始到 a 的成本
costs["b"] = 2            # 起始到 b 的成本
costs["fin"] = infinity   # 起始到 fin 的成本初始化為無窮大

# 父節點表
parents = {}
parents["a"] = "start"    # a 的父節點為 start
parents["b"] = "start"    # b 的父節點為 start
parents["fin"] = None     # fin 尚未有父節點

processed = []            # 用於追蹤已處理的節點

def find_lowest_cost_node(costs):     # 定義一個函式，用於尋找成本最低的節點
    lowest_cost = float("inf")        # 初始化最低成本為無窮大
    lowest_cost_node = None           # 初始化最低成本的節點為 None

    # 走訪所有節點
    for node in costs:
        cost = costs[node]
        # 如果目前成本低於最低成本，且尚未被處理
        if cost < lowest_cost and node not in processed:
            # 更新最低成本和最低成本的節點
            lowest_cost = cost
            lowest_cost_node = node
    return lowest_cost_node          # 傳回最低成本的節點

# 尋找尚未處理的最低成本的節點
node = find_lowest_cost_node(costs)
# 當所有節點都已處理完畢時，這個 while 循環結束
  # 逐一檢查鄰居
while node is not None:
    # 取得起點到目前節點的成本
    cost = costs[node]

    # 取得目前節點的所有鄰居
    neighbors = graph[node]

    # 逐一檢查鄰居
    for neighbor in neighbors:
        # 計算經過目前節點到達鄰居的新成本
        new_cost = cost + neighbors[neighbor]

        # 如果新路徑的成本比較低
        if new_cost < costs[neighbor]:
            # 更新成本
            costs[neighbor] = new_cost

            # 更新父節點
            parents[neighbor] = node

    # 將目前節點標示為已處理
    processed.append(node)

    # 尋找下一個尚未處理且成本最低的節點
    node = find_lowest_cost_node(costs)

print("Cost from the start to each node:")  # 輸出從起始節點到每個節點的成本
print(costs)        # 輸出成本表
