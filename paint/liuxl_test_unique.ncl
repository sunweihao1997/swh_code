begin
  ; 定义一个包含重复元素的二维数组
  a = (/(/30, 121/), (/30, 100/), (/30, 100/)/)

  ; 获取数组的大小
  n = dimsizes(a)

 ; 创建一个布尔数组来标记哪些元素是唯一的
 is_unique = new(n(0), logical)
 is_unique = True

 ; 迭代数组元素，检查重复项
 do i = 0, n(0) - 1
   if (is_unique(i)) then
     do j = i + 1, n(0) - 1
       if (all(a(i,:) .eq. a(j,:))) then
         is_unique(j) = False
       end if
     end do
   end if
 end do

 ; 计算唯一元素的数量
 unique_count = num(is_unique)

 ; 创建一个新的数组来存储唯一的元素
 unique_array = new((/unique_count, 2/), typeof(a))

 ; 将唯一的元素复制到新数组中
 idx = 0
 do i = 0, n(0) - 1
   if (is_unique(i)) then
     unique_array(idx,:) = a(i,:)
     idx = idx + 1
   end if
 end do

 ; 打印结果
 print("Original array:")
 print(a)
 print("Array with duplicates removed:")
 print(unique_array)

end