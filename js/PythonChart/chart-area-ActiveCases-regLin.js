// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

const $value_time_period_SIR = $('.value_time_period_SIR');
const $value_time_SIR = $('#range_time_period_SIR');
$value_time_period_SIR.html($value_time_SIR.val());


loadData();

$value_time_SIR.on('input change', () => {
  // loadData();
  $value_time_period_SIR.html($value_time_SIR.val());



    var url = "https://raw.githubusercontent.com/julien1941/PI/master/Python/Data/data.json?token=AL3RLGNHLEFWUOC7LHNLLRK7VJRS4"
    var data = ''
    // DAp
    var tmp ;

    $.get(url,function(data){

      const result = JSON.parse(data);



      data_sir_s = [];
      data_sir_i = [];
      data_sir_r = [];
      data_day = [];


      dataLinearfit = [];


      for(var i=0;i<$value_time_SIR.val();i++){

        data_day.push(result.current[i].Day);
        data_sir_s.push(result.current[i].SIR_S);
        data_sir_i.push(result.current[i].SIR_I);
        data_sir_r.push(result.current[i].SIR_R);


      }

  var ctx_active_cases = document.getElementById("myAreaSirModel");



  without_cum_cases(2);


  $("#num_Of_Susceptible").html((parseFloat(data_sir_s[data_sir_s.length-1]).toFixed(2)).toString())
  $("#num_Of_infected").html((parseFloat(data_sir_i[data_sir_i.length-1]).toFixed(2)).toString())
  $("#num_Of_Recovered").html((parseFloat(data_sir_r[data_sir_r.length-1]).toFixed(2)).toString())
  $("#num_Of_day").html((parseFloat(data_day[data_sir_i.length-1]).toFixed(0)).toString())
  $("#id_beta").html((parseFloat(result.parameter[0].beta).toFixed(6)).toString())
  $("#id_gamma").html((parseFloat(result.parameter[0].gamma).toFixed(6)).toString())





 },
);




});

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

function without_cum_cases(ans) {

  if (typeof(myLineChart) != "undefined"){

    myLineChart.destroy();

  }

  ctx_active_cases = document.getElementById("myAreaSirModel");
  myLineChart = new Chart(ctx_active_cases, {
    type: 'line',
    data: {
      labels: data_day,
      datasets: [
        // Susceptible
        {
          label: "Susceptible ",
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
          data: data_sir_s,
        },
        // Infectious
        {
          label: "Infectious ",
          lineTension: 0.3,
          backgroundColor: "rgba(255, 193, 7,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: data_sir_i,
        },
        // Recovered
        {
          label: "Recovered ",
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
          data: data_sir_r,
        }




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


function cum_cases(ans, dataC){
  if(ans == 2){
    return dataC;
  }
  if(ans == 1){
    return [];
  }

}

function loadData(){


    var url = "https://raw.githubusercontent.com/julien1941/PI/master/Python/Data/data.json?token=AL3RLGNHLEFWUOC7LHNLLRK7VJRS4"
    var data = ''
    // DAp
    var tmp ;

    $.get(url,function(data){

      const result = JSON.parse(data);



      data_sir_s = [];
      data_sir_i = [];
      data_sir_r = [];
      data_day = [];


      dataLinearfit = [];


      for(var i=0;i<$value_time_SIR.val();i++){

        data_day.push(result.current[i].Day);
        data_sir_s.push(result.current[i].SIR_S);
        data_sir_i.push(result.current[i].SIR_I);
        data_sir_r.push(result.current[i].SIR_R);


      }


  var ctx_active_cases = document.getElementById("myAreaSirModel");



  without_cum_cases(2);

  $("#num_Of_Susceptible").html((parseFloat(data_sir_s[data_sir_s.length-1]).toFixed(2)).toString())
  $("#num_Of_infected").html((parseFloat(data_sir_i[data_sir_i.length-1]).toFixed(2)).toString())
  $("#num_Of_Recovered").html((parseFloat(data_sir_r[data_sir_r.length-1]).toFixed(2)).toString())
  $("#num_Of_day").html((parseFloat(data_day[data_sir_i.length-1]).toFixed(0)).toString())
  $("#id_beta").html((parseFloat(result.parameter[0].beta).toFixed(6)).toString())
  $("#id_gamma").html((parseFloat(result.parameter[0].gamma).toFixed(6)).toString())




},
);}
