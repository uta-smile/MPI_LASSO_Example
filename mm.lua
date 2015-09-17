-- PUMICE Copyright (C) 2009 Lars Rosengreen (-*-coding:iso-safe-unix-*-)
-- released as free software under the terms of MIT open source license

-- import / export matrices in Matrix Market format
-- (http://math.nist.gov/MatrixMarket/formats.html)


-- require "matrix"
th = require "torch"

mm = {}


-- Import a matrix in MatrixMarket file format. 
-- Some important limitations:
--   * entries of the matrix must be real, integer or pattern (not complex)
--   * matrix must use coordinate format (not array)
--   * general, symmetric and skew-symmteric matrices are support (not
--     Hermitian)
--   * very little checking is does to ensure the integrity of the
--     import; be careful; a common pitfall is extra newlines at the
--     end of a file
function mm.import(fname)
   local file = io.open(fname, "r")
   assert(file, "unable to open file "..fname)
   local header = file:read("*l")
   -- check and make sure we can handle this Matrix Market file
   assert(string.find(header, "MatrixMarket"), 
          "Not a MatrixMarket format file")
   assert(string.find(header, "matrix"), 
          "don't know how to import that format")
   --assert(string.find(header, "coordinate"), 
   --       "matrix must be in \"coordinate\" format")
   assert(string.find(header, "integer") or string.find(header, "real") or 
          string.find(header, "pattern"), 
          "matrix type must be \"integer\", \"real\" or \"pattern\"")
   -- figure out if the matrix is supposed to be symmetric (Hermitian
   -- matrices get weeded out by assertians above), and if it is a
   -- pattern matrix (has no values, values assumed to be 1)
   local coordinate = string.find(header, "coordinate") and true
   local symmetric = string.match(header, "symmetric") and true
   local skewSymmetric = string.match(header,"skew") and true
   local pattern = string.match(header, "pattern") and true
   -- skip over comments
   for line in file:lines() do
      if line:sub(1,1) ~= "%" then
         size = line
         break
      end
   end
   local tokens = {}
   for tok in string.gmatch(size, "[%d]+") do
      tokens[#tokens + 1] = tonumber(tok)
   end
   if coordinate then 
       -- For sparse matrix
       rows, columns, nnz = tokens[1], tokens[2], tokens[3]
   else
       -- For general dense matrix
       rows, columns = tokens[1], tokens[2]
   end
   --local M = matrix.new(rows, columns)
   local counter = 0
   M = th.zeros(rows, columns)
   for line in file:lines() do
      tokens = {}
      for tok in string.gmatch(line, "[+-]?[%d\\.]+[eE]?[-+%d]*") do
         tokens[#tokens + 1] = tonumber(tok)
      end
      if coordinate then 
          row, column, value = tokens[1], tokens[2], tokens[3]
      else
          row, column = counter % rows + 1, math.floor(counter/rows)+1
          value = tokens[1]
          if column==149 and row==4 then
              print(">>"..tokens[1])
          end
      end
      counter = counter + 1
      if pattern then value = 1 end
      M[row][column] = value
      if symmetric and (row ~= column) then
         M[column][row] = (skewSymmetric and -value) or value
      end
   end
   file:close()
   return M
end


-- Export a matrix in Matrix Market "array real general" format. 
function mm.export(M, fname)
   local file = io.open(fname, "w+")
   file:write("%%MatrixMarket matrix array real general\n")
   file:write(M:size(1).." "..M:size(2).."\n")
   print(M:size(1).." "..M:size(2).."\n")
   -- Note: the matrix market format is column-oriented
   -- Therefore, the row index changes first. (strange)
   for j = 1,M:size(2) do -- columns
       for i = 1,M:size(1) do -- rows
         file:write(M[i][j].."\n")
      end
   end
   file:close()
end

--[[
function mm.export(M, fname)
   local file = io.open(fname, "w+")
   file:write("%%MatrixMarket matrix coordinate real general\n")
   file:write(M.rows.." "..M.columns.." "..M:nonzero().."\n")
   for i, v in M:vects() do
      for j, e in v:elts() do
         file:write(i.." "..j.." "..e.."\n")
      end
   end
   file:close()
end
--]]

return mm
