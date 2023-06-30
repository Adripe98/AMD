#incluimos forecast
library("forecast")

#colocamos bien el directorio de trabajo
setwd("C:/Users/Adri/Documents") 

#generamos un tipo de dato especifico para el formato de fecha del csv
setAs("character","myDate", function(from) as.Date(from, format="%d/%m/%Y") ) 

#leemos los CSV y generamos las series de todos
#temperatura media
file1<-read.csv("TM_Santiago.csv", header=TRUE, sep="|", colClasses=c("myDate","numeric"))
seriesTM_out<-ts(file1$TM, start=c(2006,1), end=c(2022, 297), frequency = 365.25)
#presion
file2<-read.csv("P_Santiago.csv", header=TRUE, sep="|", colClasses=c("myDate","numeric"))
seriesP_out<-ts(file2$P, start=c(2006,1), end=c(2022, 297), frequency = 365.25)
#irradiacion
file3<-read.csv("IRRA_Santiago.csv", header=TRUE, sep="|", colClasses=c("myDate","numeric"))
seriesIRRA_out<-ts(file3$Irradiacion.global.diaria, start=c(2006,1), end=c(2022, 297), frequency = 365.25)

#imprimimos la serie y el valor de ACF
#temperatura media
plot.ts(seriesTM_out)
acf(seriesTM_out)
#presion
plot.ts(seriesP_out)
acf(seriesP_out)
#irradiacion
plot.ts(seriesIRRA_out)
acf(seriesIRRA_out)

#limpiamos las series de outliers con la funcion TSClean de forecast
#temperatura media
seriesTM<-tsclean(seriesTM_out)
#presion
seriesP<-tsclean(seriesP_out)
#irradiacion
seriesIRRA<-tsclean(seriesIRRA_out)

#imprimimos serie y acf sin outliers
plot.ts(seriesTM)
acf(seriesTM)
plot.ts(seriesP)
acf(seriesP)
plot.ts(seriesIRRA)
acf(seriesIRRA)

#calculamos primera y segunda diferencia para todas las series (sin outliers)
seriesTM_dif1<-diff(seriesTM, differences=1)
seriesTM_dif2<-diff(seriesTM, differences=2)
seriesP_dif1<-diff(seriesP, differences=1)
seriesP_dif2<-diff(seriesP, differences=2)
seriesIRRA_dif1<-diff(seriesIRRA, differences=1)
seriesIRRA_dif2<-diff(seriesIRRA, differences=2)

#serie temporal y acf para la primera y segunda diferencia
plot.ts(seriesTM_dif1)
acf(seriesTM_dif1)
plot.ts(seriesTM_dif2)
acf(seriesTM_dif2)
plot.ts(seriesP_dif1)
acf(seriesP_dif1)
plot.ts(seriesP_dif2)
acf(seriesP_dif2)
plot.ts(seriesIRRA_dif1)
acf(seriesIRRA_dif1)
plot.ts(seriesIRRA_dif2)
acf(seriesIRRA_dif2)

#descomponemos las series en sus componentes
seriesTM_decompose<-decompose(seriesTM)
seriesP_decompose<-decompose(seriesP)
seriesIRRA_decompose<-decompose(seriesIRRA)

#visualizamos los componentes
plot(seriesTM_decompose)
plot(seriesP_decompose)
plot(seriesIRRA_decompose)

#regresion lineal para los componentes de tendencia, y visualizacion de la pendiente de los mismos
#temperatura media
linear_TM<-lm(seriesTM_decompose$trend~c(1:length(seriesTM_decompose$trend)))
plot(c(1:length(seriesTM_decompose$trend)), seriesTM_decompose$trend)
abline(linear_TM)
plot.ts(seriesTM_decompose$trend)
#presion
linear_P<-lm(seriesP_decompose$trend~c(1:length(seriesP_decompose$trend)))
plot(c(1:length(seriesP_decompose$trend)), seriesP_decompose$trend)
abline(linear_P)
plot.ts(seriesP_decompose$trend)
#irradacion
linear_IRRA<-lm(seriesIRRA_decompose$trend~c(1:length(seriesIRRA_decompose$trend)))
plot(c(1:length(seriesIRRA_decompose$trend)), seriesIRRA_decompose$trend)
abline(linear_IRRA)
plot.ts(seriesIRRA_decompose$trend)


#correlacion cruzada de las series (no estacionarias)
ccf(seriesTM, seriesIRRA)
ccf(seriesTM, seriesP)
ccf(seriesP, seriesIRRA)

#correlacion cruzada para las series una vez hechas estacionarias
ccf(seriesTM_dif2, seriesIRRA_dif2)
ccf(seriesTM_dif2, seriesP_dif2)
ccf(seriesP_dif2, seriesIRRA_dif2)

#filtrar tendencias por media movil
seriesTM_trend_media<-ma(seriesTM_decompose$trend, order=500)
plot.ts(seriesTM_trend_media)
seriesP_trend_media<-ma(seriesP_decompose$trend, order=500)
plot.ts(seriesP_trend_media)
seriesIRRA_trend_media<-ma(seriesIRRA_decompose$trend, order=500)
plot.ts(seriesIRRA_trend_media)

#predicciones, cuidado, la funcion auto.arima tarda mucho en acabar porque la serie es diaria
#y tiene que computar el modelo "optimo" para la prediccion
tm_prediccion<-auto.arima(seriesTM)
tm_prediccion
tm_prediccion_predicciones<-forecast(tm_prediccion,level=c(95),h=365)
plot(tm_prediccion_predicciones)
