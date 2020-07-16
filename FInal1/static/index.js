$(function() {
  // Make the dashboard widgets sortable Using jquery UI
  $('.connectedSortable').sortable({
    placeholder         : 'sort-highlight',
    connectWith         : '.connectedSortable',
    handle              : '.card-header, .nav-tabs',
    forcePlaceholderSize: true,
    zIndex              : 999999
  });

  $('.connectedSortable .card-header, .connectedSortable .nav-tabs-custom').css('cursor', 'move');
})