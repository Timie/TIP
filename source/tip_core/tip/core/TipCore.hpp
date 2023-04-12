#pragma once

#include <tip_core.h>

#include <type_traits>

namespace tip::core
{

    template <typename TMat1Type, typename TMat2Type, typename TOperationType>
    void applyElementwise(TMat1Type&& m1, TMat2Type&& m2, TOperationType &&operation)
    {

        typedef typename std::remove_reference<TMat1Type>::type::value_type TElem1Type;
        typedef typename std::remove_reference<TMat2Type>::type::value_type TElem2Type;

        assert(m1.size() == m2.size());

        if (m1.empty())
        {
            return;
        }

        auto rowCount = m1.rows;
        auto colCount = m1.cols;

        // If all data is continuous, we can tweak the loop a bit, to assume that all is just one huge row.
        if (m1.isContinuous() && m2.isContinuous())
        {
            colCount = colCount * rowCount;
            rowCount = 1;
        }

        for (decltype(rowCount) rowIdx = 0; rowIdx < rowCount; ++rowIdx)
        {
            auto m1Ptr = m1.template ptr<TElem1Type>(rowIdx);
            auto m2Ptr = m2.template ptr<TElem2Type>(rowIdx);

            auto endM1Ptr = m1Ptr + colCount;
            while (m1Ptr < endM1Ptr)
            {
                operation(*m1Ptr, *m2Ptr);

                ++m1Ptr;
                ++m2Ptr;
            }

        }
    }

    template <typename TMat1Type, typename TMat2Type, typename TMat3Type, typename TOperationType>
    void applyElementwise(TMat1Type&& m1, TMat2Type&& m2, TMat3Type&& m3, TOperationType &&operation)
    {

        typedef typename std::remove_reference<TMat1Type>::type::value_type TElem1Type;
        typedef typename std::remove_reference<TMat2Type>::type::value_type TElem2Type;
        typedef typename std::remove_reference<TMat3Type>::type::value_type TElem3Type;

        assert(m1.size() == m2.size());
        assert(m1.size() == m3.size());

        if (m1.empty())
        {
            return;
        }

        auto rowCount = m1.rows;
        auto colCount = m1.cols;

        // If all data is continuous, we can tweak the loop a bit, to assume that all is just one huge row.
        // TODO: Maybe update continous flags!
        if (m1.isContinuous() && m2.isContinuous() && m3.isContinuous())
        {
            colCount = colCount * rowCount;
            rowCount = 1;
        }

        for (decltype(rowCount) rowIdx = 0; rowIdx < rowCount; ++rowIdx)
        {
            auto m1Ptr = m1.template ptr<TElem1Type>(rowIdx);
            auto m2Ptr = m2.template ptr<TElem2Type>(rowIdx);
            auto m3Ptr = m3.template ptr<TElem3Type>(rowIdx);

            auto endM1Ptr = m1Ptr + colCount;
            while (m1Ptr < endM1Ptr)
            {
                operation(*m1Ptr, *m2Ptr, *m3Ptr);

                ++m1Ptr;
                ++m2Ptr;
                ++m3Ptr;
            }

        }
    }

    template <typename TMat1Type, typename TMat2Type, typename TMat3Type, typename TMat4Type, typename TOperationType>
    void applyElementwise(TMat1Type&& m1, TMat2Type&& m2, TMat3Type&& m3, TMat4Type&& m4, TOperationType &&operation)
    {

        typedef typename std::remove_reference<TMat1Type>::type::value_type TElem1Type;
        typedef typename std::remove_reference<TMat2Type>::type::value_type TElem2Type;
        typedef typename std::remove_reference<TMat3Type>::type::value_type TElem3Type;
        typedef typename std::remove_reference<TMat4Type>::type::value_type TElem4Type;

        assert(m1.size() == m2.size());
        assert(m1.size() == m3.size());
        assert(m1.size() == m4.size());

        if (m1.empty())
        {
            return;
        }

        auto rowCount = m1.rows;
        auto colCount = m1.cols;

        // If all data is continuous, we can tweak the loop a bit, to assume that all is just one huge row.
        // TODO: Maybe update continous flags!
        if (m1.isContinuous() && m2.isContinuous() && m3.isContinuous() && m4.isContinuous())
        {
            colCount = colCount * rowCount;
            rowCount = 1;
        }

        for (decltype(rowCount) rowIdx = 0; rowIdx < rowCount; ++rowIdx)
        {
            auto m1Ptr = m1.template ptr<TElem1Type>(rowIdx);
            auto m2Ptr = m2.template ptr<TElem2Type>(rowIdx);
            auto m3Ptr = m3.template ptr<TElem3Type>(rowIdx);
            auto m4Ptr = m4.template ptr<TElem4Type>(rowIdx);

            auto endM1Ptr = m1Ptr + colCount;
            while (m1Ptr < endM1Ptr)
            {
                operation(*m1Ptr, *m2Ptr, *m3Ptr, *m4Ptr);

                ++m1Ptr;
                ++m2Ptr;
                ++m3Ptr;
                ++m4Ptr;
            }

        }
    }
}