 function replaceAll(input,search,newText) {
  let result=input.split(search).join(newText);
  return result;
}
function deepCopyObj(obj) {
  let resultObj={};
  for(let o in obj) {
    resultObj[o]=obj[o];
  }
  return resultObj;
}
function json2Table(jsonObj) {
  let htmlStr="";
  let headers=[];
  let headersStored=false;
  for(var row in jsonObj) {
    var rowJsonObj=jsonObj[row];
    htmlStr+="<tr>";
    for(var colname in rowJsonObj) {
      if(!headersStored) {
        headers.push(colname);
      }
      var colval=rowJsonObj[colname];
      htmlStr+="<td>"+colval+"</td>";
    }
    htmlStr+="</tr>";
    headersStored=true;
  }
  let headersStr="<tr>";
  for(var h in headers) {
    headersStr+="<th>"+headers[h]+"</th>";
  }
  headersStr+="</tr>";
  let result=headersStr+htmlStr;
  return result;
}