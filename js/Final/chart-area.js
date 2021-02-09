// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';
const $url_data_prof = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/Cov_invaders.csv"
const $url_data_scenario_1 = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/scenario_1.csv"
const $url_data_scenario_2 = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/scenario_2.csv"
const $url_data_scenario_3 = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/scenario_3.csv"
const $url_data_scenario_4 = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/scenario_4.csv"
const $url_data_scenario_5 = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/scenario_5.csv"
const $url_data_scenario_6 = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/scenario_6.csv"
const $url_data_scenario_7 = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/scenario_7.csv"


const $value_time_period_data = $('.value_time_period_data');
// const $value_time_data = $('#range_time_period_data');

const $value_time_period_SEIR = $('.value_time_period_SEIR');
const $value_time_SEIR = $('#range_time_period_SEIR');

// $value_time_period_data.html($value_time_data.val());
$value_time_period_SEIR.html($value_time_SEIR.val());

const $id_switch_positive = document.getElementById('customSwitches_Positive');
const $id_switch_hospitalized = document.getElementById('customSwitches_hospitalized');
const $id_switch_cum_hospitalized = document.getElementById('customSwitches_cum_hospitalized');
// const $id_switch_criticals = document.getElementById('customSwitches_criticals');
// const $id_switch_fatalies = document.getElementById('customSwitches_fatalities');

const $value_customSwitchesSusceptible = $('#customSwitchesSusceptible');

const $id_switch_Susceptible = document.getElementById('customSwitchesSusceptible');
const $id_switch_Exosed = document.getElementById('customSwitchesExposed');
const $id_switch_Infected = document.getElementById('customSwitchesInfectious');
const $id_switch_Recovered = document.getElementById('customSwitchesRecovered');
const $id_switch_Hospitalized = document.getElementById('customSwitchesHospitalized');
const $id_switch_Criticals = document.getElementById('customSwitchesSeirCriticales');
const $id_switch_Death = document.getElementById('customSwitchesSeirFatalities');

// const $id_switch_num_bed_hos = document.getElementById('customSwitches_num_bed_hos');
// const $id_switch_num_bed_icu = document.getElementById('customSwitches_num_bed_icu');



$id_switch_Susceptible.addEventListener('change',function(){
    draw();
});

$id_switch_Exosed.addEventListener('change',function(){
    draw();
});

$id_switch_Infected.addEventListener('change',function(){
    draw();
});

$id_switch_Recovered.addEventListener('change',function(){
    draw();
});

$id_switch_Hospitalized.addEventListener('change',function(){
    draw();
});

$id_switch_Criticals.addEventListener('change',function(){
    draw();
});

$id_switch_Death.addEventListener('change',function(){
    draw();
});



loadData_scenario_1();
loadData_prof()


function loadData_scenario_1(){

    var data = ''
    // DAp
    var tmp ;

    $.get($url_data_scenario_1,function(data){
      var result = [];
      var lines=data.split("\n");
      var headers=lines[0].split(",");
      for(var i=1;i<lines.length;i++){
        var obj = {};
        var currentline=lines[i].split(",");
        for(var j=0;j<headers.length;j++){
          obj[headers[j]] = currentline[j];
        }
        result.push(obj);
      }


      data_day = [];
      data_S = []
      data_E = []
      data_I = []
      data_R = []
      data_H = []
      data_C = []
      data_D = []



      for(var i=0;i<$value_time_SEIR.val();i++){
        data_day.push((i).toString());
        data_S.push(result[i].S);
        data_E.push(result[i].E);
        data_I.push(result[i].I);
        data_R.push(result[i].R);
        data_H.push(result[i].H)
        data_C.push(result[i].C)
        data_D.push(result[i].D)


      }




  var ctx_active_cases = document.getElementById("myAreaSeirModel");


  // $("#id_beta_seir").html((parseFloat(result.model[0].beta).toFixed(6)).toString())
  // $("#id_sigma_seir").html((parseFloat(result.model[0].sigma).toFixed(6)).toString())
  // $("#id_gamma_seir").html((parseFloat(result.model[0].gamma).toFixed(6)).toString())
  // $("#id_hp_seir").html((parseFloat(result.model[0].hp).toFixed(6)).toString())
  // $("#id_hcr_seir").html((parseFloat(result.model[0].hcr).toFixed(6)).toString())



  // draw();



  load_card_value_seir();




},
);}


$value_time_SEIR.on('input change', () => {

  $value_time_period_SEIR.html($value_time_SEIR.val());
  loadData_scenario_1();
  loadData_prof();
});

function loadData_prof(){

    var data = ''
    // DAp
    var tmp ;

    $.get($url_data_prof,function(data){
      var result = [];
      var lines=data.split("\n");
      var headers=lines[0].split(",");
      for(var i=1;i<lines.length;i++){
        var obj = {};
        var currentline=lines[i].split(",");
        for(var j=0;j<headers.length;j++){
          obj[headers[j]] = currentline[j];
        }
        result.push(obj);
      }



      data_prof_num_positive = []
      data_prof_num_tested = []
      data_prof_num_hospitalised = []
      data_prof_num_cumulative_hospitalizations = []
      data_prof_num_critical = []
      data_prof_num_fatalities = []




      for(var i=0;i<$value_time_SEIR.val();i++){

        if (i > 191){

          data_prof_num_positive.push(0);
          data_prof_num_tested.push(0);
          data_prof_num_hospitalised.push(0);
          data_prof_num_cumulative_hospitalizations.push(0);
          data_prof_num_critical.push(0);
          data_prof_num_fatalities.push(0);
        }
        else{

          data_prof_num_positive.push(result[i].num_positive);
          data_prof_num_tested.push(result[i].num_tested);
          data_prof_num_hospitalised.push(result[i].num_hospitalised);
          data_prof_num_cumulative_hospitalizations.push(result[i].num_cumulative_hospitalizations);
          data_prof_num_critical.push(result[i].num_critical);
          data_prof_num_fatalities.push(result[i].num_fatalities);
        }


      }




  var ctx_active_cases = document.getElementById("myAreaSeirModel");



  draw();



  load_card_value_seir();




},
);}






function load_card_value_seir(){
  $("#num_Of_Susceptible_seir").html((parseFloat(data_S[data_S.length-1]).toFixed(2)).toString())
  $("#num_Of_Exposed_seir").html((parseFloat(data_E[data_E.length-1]).toFixed(2)).toString())
  $("#num_Of_infected_seir").html((parseFloat(data_I[data_I.length-1]).toFixed(2)).toString())
  $("#num_Of_Recovered_seir").html((parseFloat(data_R[data_R.length-1]).toFixed(2)).toString())


  $("#num_Of_fatalities_seir").html((parseFloat(data_D[data_D.length-1]).toFixed(2)).toString())
  $("#num_Of_day_seir").html(((parseFloat(data_day[data_day.length-1])+1).toFixed(0)).toString())

  // if($id_switch_num_bed_hos.checked == true && $id_switch_num_bed_icu.checked == true){
  //   $("#num_Of_newFat").html((parseFloat(data_seir_f_tot[data_seir_f_tot.length-1]).toFixed(0)).toString());
  //   $("#num_Of_hospitalized_seir").html((parseFloat(data_seir_h_bis[data_seir_h_bis.length-1]).toFixed(2)).toString())
  //   $("#num_Of_criticals_seir").html((parseFloat(data_seir_c_bis[data_seir_c_bis.length-1]).toFixed(2)).toString())
  // }
  // else if($id_switch_num_bed_hos.checked == false && $id_switch_num_bed_icu.checked == true){
  //   $("#num_Of_newFat").html((parseFloat(data_seir_fc_bis[data_seir_fc_bis.length-1]).toFixed(0)).toString());
  //   $("#num_Of_hospitalized_seir").html((parseFloat(data_seir_h[data_seir_h.length-1]).toFixed(2)).toString())
  //   $("#num_Of_criticals_seir").html((parseFloat(data_seir_c_bis[data_seir_c_bis.length-1]).toFixed(2)).toString())
  // }
  // else if($id_switch_num_bed_hos.checked == true && $id_switch_num_bed_icu.checked == false){
  //   $("#num_Of_newFat").html((parseFloat(data_seir_f_bis[data_seir_f_bis.length-1]).toFixed(0)).toString());
  //   $("#num_Of_criticals_seir").html((parseFloat(data_seir_c[data_seir_c.length-1]).toFixed(2)).toString())
  //   $("#num_Of_hospitalized_seir").html((parseFloat(data_seir_h_bis[data_seir_h_bis.length-1]).toFixed(2)).toString())

  // }else{
  //   $("#num_Of_newFat").html("Nan");
  //   $("#num_Of_hospitalized_seir").html((parseFloat(data_H[data_H.length-1]).toFixed(2)).toString())
  //   $("#num_Of_criticals_seir").html((parseFloat(data_C[data_C.length-1]).toFixed(2)).toString())
  // }




}



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


function positives_draw(){
  if($id_switch_positive.checked == true){
    return data_num_positive;
  }
  else{
    return [];
  }
}
function hospitalized_draw(){
  if($id_switch_hospitalized.checked == true){
      return data_H;
  }
  else{
    return [];
  }
}
function cum_hospitalized_draw(){
  if($id_switch_cum_hospitalized.checked == true){
    return data_num_cumulative_hospitalizations;
  }
  else{
    return [];
  }
}

function criticals_draw(){
  if($id_switch_Criticals.checked == true){
    return data_C;
  }
  else{
    return [];
  }
}

function fatalities_draw(){
  if($id_switch_Death.checked == true){
  // if( true){
    return data_D;
  }
  else{
    return [];
  }
}
function susceptible_draw(){
  if($id_switch_Susceptible.checked == true){
    return data_S;
  }
  else{
    return [];
  }
}
function exposed_draw(){
  if($id_switch_Exosed.checked == true){
    return data_E;
  }
  else{
    return [];
  }
}

function infected_draw(){
  if($id_switch_Infected.checked == true){
    return data_I;
  }
  else{
    return [];
  }
}

function recovered_draw(){
  if($id_switch_Recovered.checked == true){
    return data_R;
  }
  else{
    return [];
  }
}
function hospitalized_seir_draw(){
  if($id_switch_Hospitalized.checked == true){
      return data_H;
    }
  else{
    return [];
  }
}
function criticals_seir_draw(){
  if($id_switch_Criticals.checked == true){
      return data_C;
    }
  else{
    return [];
  }
}
function fatalities_seir_draw(){
  if($id_switch_Death.checked == true){
    return data_D;
  }
  else{
    return [];
  }
}



// $value_time_data.on('input change', () => {
//
//   $value_time_period_data.html($value_time_data.val());
//   load_cur_Data();
//   load_card_value();
// });

function load_card_value(){
  $("#num_day").html((parseFloat(data_day[data_day.length-1]).toFixed(0)).toString());
  $("#num_positive").html((parseFloat(data_num_positive[data_num_positive.length-1]).toFixed(0)).toString());
  $("#num_tested").html((parseFloat(data_num_tested[data_num_tested.length-1]).toFixed(0)).toString());
  $("#num_hospitalised").html((parseFloat(data_num_hospitalised[data_num_hospitalised.length-1]).toFixed(0)).toString());
  $("#num_cumulative_hospitalizations").html((parseFloat(data_num_cumulative_hospitalizations[data_num_cumulative_hospitalizations.length-1]).toFixed(0)).toString());
  $("#num_critical").html((parseFloat(data_num_critical[data_num_critical.length-1]).toFixed(0)).toString());
  $("#num_fatalities").html((parseFloat(data_num_fatalities[data_num_fatalities.length-1]).toFixed(0)).toString());




}
function load_card_kpi_value(){


  var percentChange_positive = data_num_positive[data_num_positive.length-1]-data_num_positive[data_num_positive.length-2]
  var percentChange_hos = data_num_hospitalised[data_num_hospitalised.length-1]-data_num_positive[data_num_hospitalised.length-2]
  var percentChange_cum_hos = data_num_hospitalised[data_num_cumulative_hospitalizations.length-1]-data_num_positive[data_num_cumulative_hospitalizations.length-2]
  var percentChange_crit = data_num_critical[data_num_critical.length-1]-data_num_critical[data_num_critical.length-2]
  var percentChange_fat = data_num_fatalities[data_num_fatalities.length-1]-data_num_fatalities[data_num_fatalities.length-2]

  if(percentChange_positive == 0){
    result = `
    <td class="fas fa-caret-right text-success">${ percentChange_positive }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#casesKpi").html(result)
  }
  else if(percentChange_positive > 0){
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_positive }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#casesKpi").html(result)
  }
  else{
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_positive }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#casesKpi").html(result)
  }


  if(percentChange_hos == 0){
    result = `
    <td class="fas fa-caret-right text-success">${ percentChange_hos }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_hospitalisedKpi").html(result)
  }
  else if(percentChange_hos < 0){
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_hos }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_hospitalisedKpi").html(result)
  }
  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_hos }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_hospitalisedKpi").html(result)
  }

    if(percentChange_cum_hos == 0){
      result = `
      <td class="fas fa-caret-right text-success">${ percentChange_cum_hos }</td>
      <td class="fas fa-percent text-success"></td>`
      $("#num_cmulative_hospitalizationsKpi").html(result)
    }
    else if(percentChange_cum_hos < 0){
      result = `
      <td class="fas fa-caret-down fa-1x text-success">${ percentChange_cum_hos }</td>
      <td class="fas fa-percent text-success"></td>`
      $("#num_cmulative_hospitalizationsKpi").html(result)
    }
    else{
      result = `
      <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_cum_hos }</td>
      <td class="fas fa-percent text-danger"></td>`
      $("#num_cmulative_hospitalizationsKpi").html(result)
    }

  if(percentChange_crit == 0){
    result = `
    <td class="fas fa-caret-right fa-1x text-success">${ percentChange_crit }</td>
    <td class="fas fa-percent text-success"></td>`

    $("#num_criticalKpi").html(result)
  }
  else if(percentChange_crit < 0){
    result = `


    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_crit }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_criticalKpi").html(result)
  }

  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_crit }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_criticalKpi").html(result)
  }
  if(percentChange_fat == 0){
    result = `
    <td class="fas fa-caret-right fa-1x text-success">${ percentChange_fat }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_fatalitiesKpi").html(result)
  }
  else if(percentChange_fat < 0){
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_fat }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_fatalitiesKpi").html(result)
  }

  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_fat  }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_fatalitiesKpi").html(result)
  }




}


function draw() {

  if (typeof(myLineChart) != "undefined"){

    myLineChart.destroy();

  }


  ctx_active_cases = document.getElementById("myAreaSeirModel").getContext("2d");;
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
          pointRadius: 1,
          pointBackgroundColor: "rgba(78, 115, 223, 1)",
          pointBorderColor: "rgba(78, 115, 223, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
          pointHoverBorderColor: "rgba(78, 115, 223, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: susceptible_draw(),
        },
        // Exposed
        {
          label: "Exposed ",
          lineTension: 0.6,
          backgroundColor: "rgba(240, 173, 78, 0.2)",
          borderColor: "rgba(240, 173, 78, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(240, 173, 78, 1)",
          pointBorderColor: "rgba(240, 173, 78, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(240, 173, 78, 1)",
          pointHoverBorderColor: "rgba(240, 173, 78, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: exposed_draw(),
        },
        // Infectious
        {
          label: "Infectious ",
          lineTension: 0.3,
          backgroundColor: "rgba(237, 0, 59,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: infected_draw(),
        },
        // Recovered
        {
          label: "Recovered ",
          lineTension: 0.6,
          backgroundColor: "rgba(37, 56, 60, 0.2)",
          borderColor: "rgba(37, 56, 60, 0.1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(37, 56, 60, 0.1)",
          pointBorderColor: "rgba(37, 56, 60, 0.1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(37, 56, 60, 0.1)",
          pointHoverBorderColor: "rgba(37, 56, 60, 0.1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: recovered_draw(),
        },
        // Hospitalized
        {
          label: "hospitalized ",
          lineTension: 0.6,
          backgroundColor: "rgba(34,139,34, 0.2)",
          borderColor: "rgba(34,139,34, 0.1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(34,139,34, 0.1)",
          pointBorderColor: "rgba(34,139,34, 0.1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(34,139,34, 0.1)",
          pointHoverBorderColor: "rgba(34,139,34, 0.1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: hospitalized_seir_draw(),
        },

        // Criticals
        {
          label: "Criticals ",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.3)",
          borderColor: "rgba(255, 193, 7,1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(255, 193, 7,1)",
          pointBorderColor: "rgba(255, 193, 7,1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(255, 193, 7, 1)",
          pointHoverBorderColor: "rgba(255, 193, 7, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: criticals_seir_draw(),
        },
        // Fatalities
        {
          label: "Fatalities",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: fatalities_seir_draw(),
        },
        // Plot data prof
        {
          label: "data_prof_num_hospitalised",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0,0.1)",
          borderColor: "rgba(0, 0, 0, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(0, 0, 0, 1)",
          pointBorderColor: "rgba(0, 0, 0, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
          pointHoverBorderColor: "rgba(0, 0, 0, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: data_prof_num_hospitalised,
        },
        {
          label: "data_prof_num_critical",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0,0.1)",
          borderColor: "rgba(0, 0, 0, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(0, 0, 0, 1)",
          pointBorderColor: "rgba(0, 0, 0, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
          pointHoverBorderColor: "rgba(0, 0, 0, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: data_prof_num_critical,
        },
        {
          label: "data_prof_num_fatalities",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0,0.1)",
          borderColor: "rgba(0, 0, 0, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(0, 0, 0, 1)",
          pointBorderColor: "rgba(0, 0, 0, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
          pointHoverBorderColor: "rgba(0, 0, 0, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: data_prof_num_fatalities,
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
