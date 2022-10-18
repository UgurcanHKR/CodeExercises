using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ml_pred_ConsoleApp1.Schemas
{
    internal class LoadParameters
    {
        [LoadColumn(0)]
        public float DAY7;

        [LoadColumn(1)]
        public float DAY2;

        [LoadColumn(2)]
        public float DAYF;
    }
}
