- log transofrmation cols:

- root transfomration 

margin_gross_pow_ele
margin_net_pow_ele
net_margin

![img](https://www.google.com/url?sa=i&url=https%3A%2F%2Fstats.stackexchange.com%2Fquestions%2F107610%2Fwhat-is-the-reason-the-log-transformation-is-used-with-right-skewed-distribution&psig=AOvVaw0LG7GrFJLPsuyil-cY4SRe&ust=1723461366314000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCNCwz4To7IcDFQAAAAAdAAAAABA4)

[img1]()

- bimodal 


'cons_12m': log10


col	skewness	kurtosis	kstest_stat	kstest_p	adtest_stat	adtest_siglevel

10	net_margin	36.569515	2642.965291	0.271922	0.0	1490.288291	5.0 log10 **std0.55**

27	var_6m_price_off_peak_fix	22.886924	540.613880	0.479801	0.0	5530.705609	5.0
30	var_6m_price_off_peak	22.886911	540.613340	0.479801	0.0	5530.696300	5.0
18	var_year_price_off_peak_fix	22.051551	521.803464	0.475041	0.0	5412.108786	5.0
21	var_year_price_off_peak	22.051457	521.799644	0.475039	0.0	5412.083581	5.0
24	var_6m_price_off_peak_var	16.954975	323.580562	0.473724	0.0	5056.296447	5.0

4	forecast_cons_year	16.587990	653.734407	0.333238	0.0	2027.996168	5.0 log10 *std 1.58*

3	imp_cons	13.198799	380.893698	0.327232	0.0	1949.763939	5.0 log10 *std 1.14*

15	var_year_price_off_peak_var	12.052339	179.055548	0.437688	0.0	4605.389315	5.0 cbrt+log1p **std0.01** 
31	var_6m_price_peak	10.721069	139.195498	0.525013	0.0	5533.544074	5.0
28	var_6m_price_peak_fix	10.721053	139.195356	0.525013	0.0	5533.574896	5.0
26	var_6m_price_mid_peak_var	10.072325	108.059986	0.513584	0.0	5466.973436	5.0
29	var_6m_price_mid_peak_fix	9.862636	101.659456	0.519896	0.0	5535.152874	5.0
32	var_6m_price_mid_peak	9.862635	101.659446	0.519883	0.0	5535.150666	5.0
1	cons_gas_12m	9.597530	126.333634	0.431791	0.0	4746.616730	5.0
16	var_year_price_peak_var	8.979052	117.720619	0.469609	0.0	4613.129931	5.0
25	var_6m_price_peak_var	8.948396	97.994361	0.503719	0.0	5229.898567	5.0
17	var_year_price_mid_peak_var	8.720249	99.413499	0.504765	0.0	5106.395355	5.0
22	var_year_price_peak	7.908207	73.999070	0.528726	0.0	5288.390547	5.0
19	var_year_price_peak_fix	7.908174	73.998497	0.528734	0.0	5288.417630	5.0
23	var_year_price_mid_peak	7.907295	71.796921	0.524511	0.0	5288.114371	5.0
20	var_year_price_mid_peak_fix	7.907288	71.796753	0.524527	0.0	5288.120774	5.0

5	forecast_cons_12m	7.155853	147.426681	0.216919	0.0	1116.653803	5.0 log10 **std0.68**

2	cons_last_month	6.391407	47.762991	0.409264	0.0	4130.556273	5.0 log 10 *std 1.77*

0	cons_12m	5.997308	42.689777	0.406488	0.0	3987.268896	5.0 Log10 **std 0.88**

11	pow_max	5.786785	59.202563	0.279019	0.0	1910.812866	5.0 reciprocal **std 0.02(left skew)**

~~6	forecast_discount_energy	5.155098	24.854712	0.539996	0.0	5369.906870	5.0~~ *not normally distributed* -> transfomr to bool 

9	margin_net_pow_ele	4.473326	35.901232	0.169513	0.0	789.110856	5.0 log10 **std 0.34** (DROPPED)
8	margin_gross_pow_ele	4.472632	35.892607	0.169497	0.0	789.021890	5.0 log10 **std 0.34**

7	forecast_meter_rent_12m	1.505148	4.491521	0.305924	0.0	1550.621492	5.0 log10 **std 0.57**

13	forecast_price_energy_peak	-0.014331	-1.890755	0.329107	0.0	2082.971719	5.0 log10 **std 0.02**

12	forecast_price_energy_off_peak	-0.119586	8.364539	0.166637	0.0	635.598202	5.0  log10 **std 0.01**
14	forecast_price_pow_off_peak	-4.998772	54.708041	0.280248	0.0	1762.928313	5.0 log10 **std 0.13** but the skew is higher than original 