// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';
const $url_data_19 = "https://epistat.sciensano.be/Data/COVID19BE_CASES_AGESEX.json"


loadData();


function loadData(){

    var data_cov19_be = ''
    // DAp
    var tmp ;

    $.get($url_data_19,function(data_cov19_be){

      const result = JSON.parse(data_cov19_be);

      console.log(result)
      data_day_seir = [];
      data_seir_s = [];
      data_seir_e = [];
      data_seir_i = [];
      data_seir_r = [];
      data_seir_h = [];
      data_seir_c = [];
      data_seir_f = [];


      // for(var i=0;i<$value_time_SEIR.val();i++){
      //
      //   data_day_seir.push(result.predict[i].predict_day);
      //   data_seir_s.push(result.predict[i].predict_S);
      //   data_seir_e.push(result.predict[i].predict_E);
      //   data_seir_i.push(result.predict[i].predict_I);
      //   data_seir_r.push(result.predict[i].predict_R);
      //   data_seir_h.push(result.predict[i].predict_H);
      //   data_seir_c.push(result.predict[i].predict_C);
      //   data_seir_f.push(result.predict[i].predict_F);
      //
      //
      //
      // }


  // var ctx_active_cases19 = document.getElementById("myAreaSeirModel");
  // draw_current_data19();








},
);}

// $value_time_SEIR.on('input change', () => {
//
//   $value_time_period_SEIR.html($value_time_SEIR.val());
//   loadData();
// });


function number_format(number, decimals, dec_point, thousands_sep) {
  // *     example: number_format(1234.56, 2, ',', ' ');
  // *     return: '1 234,56'
  number = (number + '').replace(',', '').replace(' ', '');
  var n = !isFinite(+number) ? 0 : +number,
    prec = !isFinite(+decimals) ? 0 : Math.abs(decimals),
    sep = (typeof thousands_sep === 'undefined') ? ',' : thousands_sep,
    dec = (typeof dec_point === 'undefined') ? '.' : dec_point,
    s = '',
    toFixedFix = function(n, prec) {
      var k = Math.pow(10, prec);
      return '' + Math.round(n * k) / k;
    };
  // Fix for IE parseFloat(0.55).toFixed(0) = 0;
  s = (prec ? toFixedFix(n, prec) : '' + Math.round(n)).split('.');
  if (s[0].length > 3) {
    s[0] = s[0].replace(/\B(?=(?:\d{3})+(?!\d))/g, sep);
  }
  if ((s[1] || '').length < prec) {
    s[1] = s[1] || '';
    s[1] += new Array(prec - s[1].length + 1).join('0');
  }
  return s.join(dec);
}


// function positives_draw(){
//   if($id_switch_positive19.checked == true){
//     return data_num_positive19;
//   }
//   else{
//     return [];
//   }
// }
// function hospitalized_draw(){
//   if($id_switch_hospitalized.checked == true){
//     return data_num_hospitalised;
//   }
//   else{
//     return [];
//   }
// }
// function cum_hospitalized_draw(){
//   if($id_switch_cum_hospitalized.checked == true){
//     return data_num_cumulative_hospitalizations;
//   }
//   else{
//     return [];
//   }
// }
//
// function criticals_draw(){
//   if($id_switch_criticals.checked == true){
//     return data_num_critical;
//   }
//   else{
//     return [];
//   }
// }
//
// function fatalities_draw(){
//   if($id_switch_fatalies.checked == true){
//     return data_num_fatalities;
//   }
//   else{
//     return [];
//   }
// }
// function susceptible_draw(){
//   if($id_switch_Susceptible.checked == true){
//     return data_seir_s;
//   }
//   else{
//     return [];
//   }
// }
// function exposed_draw(){
//   if($id_switch_Exosed.checked == true){
//     return data_seir_e;
//   }
//   else{
//     return [];
//   }
// }
//
// function infected_draw(){
//   if($id_switch_Infected.checked == true){
//     return data_seir_i;
//   }
//   else{
//     return [];
//   }
// }
//
// function recovered_draw(){
//   if($id_switch_Recovered.checked == true){
//     return data_seir_r;
//   }
//   else{
//     return [];
//   }
// }
// function hospitalized_seir_draw(){
//   if($id_switch_Hospitalized.checked == true){
//     return data_seir_h;
//   }
//   else{
//     return [];
//   }
// }
// function criticals_seir_draw(){
//   if($id_switch_seir_criticals.checked == true){
//     return data_seir_c;
//   }
//   else{
//     return [];
//   }
// }
// function fatalities_seir_draw(){
//   if($id_switch_seir_fatalities.checked == true){
//     return data_seir_f;
//   }
//   else{
//     return [];
//   }
// }
//
// $value_time_data.on('input change', () => {
//
//   $value_time_period_data.html($value_time_data.val());
//
//
// });
//
//
//
function draw_current_data19() {

  if (typeof(myLineChart_positive) != "undefined"){

    myLineChart_positive.destroy();

  }

  ctx_positive = document.getElementById("myAreaChartPositive");
  myLineChart_positive = new Chart(ctx_positive, {
    type: 'line',
    data: {
      labels: data_day,
      datasets: [
        // positives
        {
          label: "Positives ",
          lineTension: 0.6,
          backgroundColor: "rgba(78, 115, 223, 0.2)",
          borderColor: "rgba(78, 115, 223, 1)",
          pointRadius: 4,
          pointBackgroundColor: "rgba(78, 115, 223, 1)",
          pointBorderColor: "rgba(78, 115, 223, 1)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
          pointHoverBorderColor: "rgba(78, 115, 223, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 4,
          data: positives_draw(),
        },
        // Hospitalised
        {
          label: "Hospitalised ",
          lineTension: 0.6,
          backgroundColor: "rgba(34,139,34, 0.2)",
          borderColor: "rgba(34,139,34, 0.1)",
          pointRadius: 4,
          pointBackgroundColor: "rgba(34,139,34, 0.1)",
          pointBorderColor: "rgba(34,139,34, 0.1)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba(34,139,34, 0.1)",
          pointHoverBorderColor: "rgba(34,139,34, 0.1)",
          pointHitRadius: 10,
          pointBorderWidth: 4,
          data: hospitalized_draw(),
        },
        // Hospitalised
        {
          label: "Hospitalised ",
          lineTension: 0.6,
          backgroundColor: "rgba(34,139,34, 0.2)",
          borderColor: "rgba(34,139,34, 0.1)",
          pointRadius: 4,
          pointBackgroundColor: "rgba(34,139,34, 0.1)",
          pointBorderColor: "rgba(34,139,34, 0.1)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba(34,139,34, 0.1)",
          pointHoverBorderColor: "rgba(34,139,34, 0.1)",
          pointHitRadius: 10,
          pointBorderWidth: 4,
          data: cum_hospitalized_draw(),
        },
        // Criticals
        {
          label: "criticals ",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.3)",
          borderColor: "rgba(255, 193, 7,1)",
          pointRadius: 4,
          pointBackgroundColor: "rgba(255, 193, 7,1)",
          pointBorderColor: "rgba(255, 193, 7,1)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba(255, 193, 7, 1)",
          pointHoverBorderColor: "rgba(255, 193, 7, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: criticals_draw(),
        },
        // Fatalies
        {
          label: "Fatalies ",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 4,
          data: fatalities_draw(),
        },






    ],
    },
    options: {
      maintainAspectRatio: false,
      layout: {
        padding: {
          left: 10,
          right: 25,
          top: 25,
          bottom: 0
        }
      },
      scales: {
        xAxes: [{
          time: {
            unit: 'date'
          },
          gridLines: {
            display: false,
            drawBorder: false
          },
          ticks: {
            maxTicksLimit: 7,
            // callback: function(value, index, values) {
            //   return number_format(value)+ ' Days';
            // }
          }
        }],
        yAxes: [{
          ticks: {
            maxTicksLimit: 5,
            padding: 10,

            callback: function(value, index, values) {
              return number_format(value)+ ' Cases';
            }
          },
          gridLines: {
            color: "rgb(234, 236, 244)",
            zeroLineColor: "rgb(234, 236, 244)",
            drawBorder: false,
            borderDash: [2],
            zeroLineBorderDash: [2]
          }
        }],
      },
      legend: {
        display: false
      },
      tooltips: {
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        titleMarginBottom: 10,
        titleFontColor: '#6e707e',
        titleFontSize: 14,
        borderColor: '#dddfeb',
        borderWidth: 1,
        xPadding: 15,
        yPadding: 15,
        displayColors: false,
        intersect: false,
        mode: 'index',
        caretPadding: 10,
        callbacks: {
          label: function(tooltipItem, chart) {
            var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
            return datasetLabel + number_format(tooltipItem.yLabel)+ ': Cases';
          }
        }
      }
    }
  },);


}
