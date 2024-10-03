select *
from transaction
limit 10;


select product_id, sum(quantity) as total_pembelian_per_produk 
from transaction 
group by product_id 
order by sum(quantity) desc;


select product_name,category,sub_category
from ref_product 
where product_id = (
	select product_id from transaction 
    group by product_id 
    order by sum(profit) desc 
    limit 1
);

SELECT a.customer_id,a.customer_name, round(sum(b.sales)) as total_pembelian
FROM ref_customer as a
JOIN transaction as b
ON a.customer_id = b.customer_id
GROUP BY a.customer_id,a.customer_name;